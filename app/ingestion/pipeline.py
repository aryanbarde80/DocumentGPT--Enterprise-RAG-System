"""
Ingestion pipeline — orchestrates parsing → chunking → embedding → storage.
Designed for async, concurrent processing of large document batches.
"""
import asyncio
import time
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.core.models import DocumentMetadata, FileType, IngestResponse, ParsedDocument
from app.ingestion.chunker import ParentChildChunker
from app.ingestion.parsers import ParserFactory
from app.vectorstore.embedder import EmbeddingService
from app.vectorstore.pinecone_store import PineconeVectorStore

logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.
    Thread-safe, async, supports concurrent document processing.
    """

    def __init__(
        self,
        embedder: EmbeddingService,
        vector_store: PineconeVectorStore,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunker = ParentChildChunker()
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_INGEST)

    async def ingest_file(
        self,
        file_path: Path,
        file_name: str,
        namespace: str = "default",
        custom_metadata: dict | None = None,
    ) -> IngestResponse:
        """
        Full pipeline for a single document:
        parse → normalize → chunk → embed → store → respond
        """
        async with self._semaphore:
            start_ts = time.monotonic()

            try:
                # 1. Detect file type and parse
                file_type = ParserFactory.get_file_type(file_name)
                parser = ParserFactory.get_parser(file_type)

                logger.info("ingestion_start", file=file_name, file_type=file_type)
                raw_text, page_count = await parser.parse(file_path)

                if not raw_text.strip():
                    raise IngestionError(f"No text content extracted from {file_name}")

                # 2. Build metadata
                meta = DocumentMetadata(
                    source_file=file_name,
                    file_type=file_type,
                    file_size_bytes=file_path.stat().st_size,
                    page_count=page_count,
                    namespace=namespace,
                    custom_metadata=custom_metadata or {},
                )

                # 3. Chunk document (parent-child)
                parsed_doc: ParsedDocument = self._chunker.chunk(raw_text, meta)

                # 4. Generate embeddings for child chunks (for retrieval)
                # Parent chunks stored as metadata, NOT embedded directly
                child_contents = [c.content for c in parsed_doc.child_chunks]
                child_embeddings = await self._embedder.embed_batch(child_contents)

                for chunk, embedding in zip(parsed_doc.child_chunks, child_embeddings):
                    chunk.embedding = embedding

                # 5. Upsert to Pinecone (child chunks carry parent content in metadata)
                parent_map = {p.chunk_id: p.content for p in parsed_doc.parent_chunks}
                await self._vector_store.upsert_chunks(
                    chunks=parsed_doc.child_chunks,
                    parent_map=parent_map,
                    namespace=namespace,
                )

                elapsed_ms = (time.monotonic() - start_ts) * 1000
                logger.info(
                    "ingestion_complete",
                    file=file_name,
                    doc_id=meta.doc_id,
                    parent_chunks=len(parsed_doc.parent_chunks),
                    child_chunks=len(parsed_doc.child_chunks),
                    elapsed_ms=round(elapsed_ms, 2),
                )

                return IngestResponse(
                    doc_id=meta.doc_id,
                    status="success",
                    file_name=file_name,
                    parent_chunks=len(parsed_doc.parent_chunks),
                    child_chunks=len(parsed_doc.child_chunks),
                    ingestion_time_ms=round(elapsed_ms, 2),
                    namespace=namespace,
                )

            except IngestionError:
                raise
            except Exception as exc:
                logger.error("ingestion_failed", file=file_name, error=str(exc))
                raise IngestionError(
                    f"Ingestion failed for {file_name}: {exc}"
                ) from exc

    async def ingest_batch(
        self,
        files: list[tuple[Path, str]],
        namespace: str = "default",
    ) -> list[IngestResponse]:
        """Concurrently ingest a batch of (path, name) tuples."""
        tasks = [
            self.ingest_file(path, name, namespace=namespace)
            for path, name in files
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        responses = []
        for name_tuple, result in zip(files, results):
            if isinstance(result, Exception):
                logger.error("batch_ingest_item_failed", file=name_tuple[1], error=str(result))
            else:
                responses.append(result)
        return responses
