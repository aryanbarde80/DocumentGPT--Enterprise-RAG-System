"""
Hybrid retrieval — fuses dense (Pinecone) and sparse (BM25) results.

Fusion formula:
  hybrid_score = α * dense_score + (1 - α) * sparse_score
  where α = HYBRID_DENSE_WEIGHT (default: 0.7)

After fusion, parent chunks are expanded to give the LLM broader context.
Results are deduplicated and token-budget-aware.
"""
import asyncio
import time
from collections import defaultdict

import tiktoken

from app.core.config import settings
from app.core.logging import get_logger
from app.core.models import ChunkType, RetrievedChunk, RetrievalResult
from app.retrieval.sparse_retriever import BM25SparseRetriever
from app.vectorstore.embedder import EmbeddingService
from app.vectorstore.pinecone_store import PineconeVectorStore

logger = get_logger(__name__)
_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


class HybridRetriever:
    """
    Two-stage hybrid retrieval:
      1. Dense search via Pinecone (semantic similarity)
      2. Sparse search via BM25 (keyword overlap)
      3. Score fusion with configurable α
      4. Parent context expansion
      5. Token-budget-aware context assembly
    """

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        embedder: EmbeddingService,
        sparse_retriever: BM25SparseRetriever,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._sparse = sparse_retriever
        self._dense_weight = settings.HYBRID_DENSE_WEIGHT
        self._sparse_weight = settings.HYBRID_SPARSE_WEIGHT

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        namespace: str = "default",
    ) -> RetrievalResult:
        start_ts = time.monotonic()

        # 1. Embed query (cached)
        query_embedding = await self._embedder.embed_one(query)

        # 2. Dense + sparse search in parallel
        dense_task = self._vector_store.query(
            embedding=query_embedding,
            top_k=settings.PINECONE_TOP_K,
            namespace=namespace,
        )
        sparse_results_sync = self._sparse.query(query, top_k=settings.PINECONE_TOP_K)
        dense_results, _ = await asyncio.gather(dense_task, asyncio.sleep(0))
        sparse_results = sparse_results_sync  # BM25 is sync

        # 3. Fuse scores
        fused = self._fuse_scores(dense_results, sparse_results)

        # 4. Sort by hybrid score
        ranked = sorted(fused.values(), key=lambda x: x["hybrid_score"], reverse=True)

        # 5. Expand child → parent for top candidates
        expanded = self._expand_to_parent(ranked)

        # 6. Deduplicate and enforce token budget
        final_chunks = self._apply_token_budget(
            expanded, max_tokens=settings.MAX_CONTEXT_TOKENS, top_k=top_k
        )

        elapsed_ms = (time.monotonic() - start_ts) * 1000
        total_tokens = sum(_count_tokens(c.content) for c in final_chunks)

        logger.info(
            "retrieval_complete",
            query=query[:60],
            dense_hits=len(dense_results),
            sparse_hits=len(sparse_results),
            final_chunks=len(final_chunks),
            total_tokens=total_tokens,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            total_tokens=total_tokens,
            retrieval_latency_ms=round(elapsed_ms, 2),
        )

    def _fuse_scores(
        self,
        dense_results: list[dict],
        sparse_results: list[tuple[str, float]],
    ) -> dict[str, dict]:
        """Merge dense and sparse scores into a unified score map."""
        fused: dict[str, dict] = {}

        # Index sparse results for O(1) lookup
        sparse_map = dict(sparse_results)

        for item in dense_results:
            cid = item["chunk_id"]
            dense_score = item["score"]
            sparse_score = sparse_map.get(cid, 0.0)
            hybrid_score = (
                self._dense_weight * dense_score
                + self._sparse_weight * sparse_score
            )
            fused[cid] = {
                "chunk_id": cid,
                "metadata": item["metadata"],
                "dense_score": dense_score,
                "sparse_score": sparse_score,
                "hybrid_score": hybrid_score,
            }

        # Include sparse-only hits not in dense results
        for cid, sparse_score in sparse_results:
            if cid not in fused:
                fused[cid] = {
                    "chunk_id": cid,
                    "metadata": {},
                    "dense_score": 0.0,
                    "sparse_score": sparse_score,
                    "hybrid_score": self._sparse_weight * sparse_score,
                }

        return fused

    def _expand_to_parent(self, ranked: list[dict]) -> list[dict]:
        """
        For each ranked child chunk, prefer its parent content when available.
        Deduplicates by parent_id so we don't include the same parent twice.
        """
        seen_parents: set[str] = set()
        expanded: list[dict] = []

        for item in ranked:
            meta = item.get("metadata", {})
            parent_id = meta.get("parent_id", "")
            parent_content = meta.get("parent_content", "")
            child_content = meta.get("content", "")

            # Use parent content if available (richer context)
            if parent_id and parent_content:
                if parent_id in seen_parents:
                    continue  # Already included this parent
                seen_parents.add(parent_id)
                item["display_content"] = parent_content
            else:
                item["display_content"] = child_content

            expanded.append(item)

        return expanded

    def _apply_token_budget(
        self,
        ranked: list[dict],
        max_tokens: int,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Build final chunk list respecting token budget and top_k."""
        chunks: list[RetrievedChunk] = []
        used_tokens = 0

        for item in ranked:
            if len(chunks) >= top_k:
                break

            meta = item.get("metadata", {})
            content = item.get("display_content", meta.get("content", ""))
            token_count = _count_tokens(content)

            if used_tokens + token_count > max_tokens and chunks:
                # Truncate last chunk to fit budget if needed
                break

            used_tokens += token_count
            chunks.append(
                RetrievedChunk(
                    chunk_id=item["chunk_id"],
                    doc_id=meta.get("doc_id", ""),
                    parent_id=meta.get("parent_id") or None,
                    chunk_type=ChunkType(meta.get("chunk_type", "child")),
                    content=content,
                    source_file=meta.get("source_file", "unknown"),
                    dense_score=item["dense_score"],
                    sparse_score=item["sparse_score"],
                    hybrid_score=item["hybrid_score"],
                    metadata=meta,
                )
            )

        return chunks
