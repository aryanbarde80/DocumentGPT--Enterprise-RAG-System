"""
Pinecone vector store — async wrapper with upsert batching,
namespace support, and structured metadata storage.
"""
import asyncio
from typing import Any, Optional

from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.core.models import DocumentChunk, RetrievedChunk, ChunkType

logger = get_logger(__name__)

# Pinecone upsert batch limit
_UPSERT_BATCH_SIZE = 100


def _chunk_to_vector(
    chunk: DocumentChunk,
    parent_map: dict[str, str],
) -> dict[str, Any]:
    """Convert a DocumentChunk to a Pinecone vector record."""
    parent_content = parent_map.get(chunk.parent_id, "") if chunk.parent_id else chunk.content
    return {
        "id": chunk.chunk_id,
        "values": chunk.embedding,
        "metadata": {
            "doc_id": chunk.doc_id,
            "parent_id": chunk.parent_id or "",
            "chunk_type": chunk.chunk_type.value,
            "content": chunk.content,
            "parent_content": parent_content[:4000],  # Pinecone metadata limit
            "source_file": chunk.metadata.source_file,
            "file_type": chunk.metadata.file_type.value,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "namespace": chunk.metadata.namespace,
        },
    }


class PineconeVectorStore:
    """
    Production Pinecone client.
    - Auto-creates index if missing
    - Batched upserts for throughput
    - Async via thread executor (Pinecone SDK is sync)
    """

    def __init__(self) -> None:
        self._pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._index_name = settings.PINECONE_INDEX_NAME
        self._index = None

    async def initialize(self) -> None:
        """Create index if it doesn't exist, then connect."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_initialize)
        logger.info("pinecone_initialized", index=self._index_name)

    def _sync_initialize(self) -> None:
        existing = [idx.name for idx in self._pc.list_indexes()]
        if self._index_name not in existing:
            logger.info("pinecone_creating_index", index=self._index_name)
            self._pc.create_index(
                name=self._index_name,
                dimension=settings.EMBEDDING_DIMENSIONS,
                metric=settings.PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.PINECONE_ENVIRONMENT,
                ),
            )
        self._index = self._pc.Index(self._index_name)

    def _get_index(self):
        if self._index is None:
            self._index = self._pc.Index(self._index_name)
        return self._index

    async def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        parent_map: dict[str, str],
        namespace: str = "default",
    ) -> None:
        """Batch upsert child chunks to Pinecone."""
        if not chunks:
            return

        vectors = [_chunk_to_vector(c, parent_map) for c in chunks if c.embedding]
        if not vectors:
            raise VectorStoreError("No embeddings found on chunks for upsert")

        loop = asyncio.get_event_loop()

        async def _upsert_batch(batch: list[dict]) -> None:
            await loop.run_in_executor(
                None,
                lambda: self._get_index().upsert(vectors=batch, namespace=namespace),
            )

        tasks = []
        for i in range(0, len(vectors), _UPSERT_BATCH_SIZE):
            batch = vectors[i : i + _UPSERT_BATCH_SIZE]
            tasks.append(_upsert_batch(batch))

        await asyncio.gather(*tasks)
        logger.info(
            "pinecone_upsert_complete",
            count=len(vectors),
            namespace=namespace,
        )

    async def query(
        self,
        embedding: list[float],
        top_k: int = 20,
        namespace: str = "default",
        filter_dict: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """Query Pinecone for nearest neighbors."""
        loop = asyncio.get_event_loop()

        def _sync_query() -> list[dict]:
            index = self._get_index()
            kwargs: dict[str, Any] = {
                "vector": embedding,
                "top_k": top_k,
                "namespace": namespace,
                "include_metadata": True,
            }
            if filter_dict:
                kwargs["filter"] = filter_dict

            result = index.query(**kwargs)
            return [
                {
                    "chunk_id": match["id"],
                    "score": match["score"],
                    "metadata": match.get("metadata", {}),
                }
                for match in result.get("matches", [])
            ]

        try:
            return await loop.run_in_executor(None, _sync_query)
        except Exception as exc:
            raise VectorStoreError(f"Pinecone query failed: {exc}") from exc

    async def delete_by_doc_id(self, doc_id: str, namespace: str = "default") -> None:
        """Delete all vectors for a given document."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._get_index().delete(
                filter={"doc_id": {"$eq": doc_id}},
                namespace=namespace,
            ),
        )
        logger.info("pinecone_deleted", doc_id=doc_id)

    async def health_check(self) -> bool:
        """Verify Pinecone connectivity."""
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                None, lambda: self._get_index().describe_index_stats()
            )
            return stats is not None
        except Exception:
            return False
