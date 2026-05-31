"""
Sparse retrieval engine using BM25.

Maintains an in-memory BM25 corpus indexed by chunk_id.
For large deployments (>500k chunks), replace with Elasticsearch or OpenSearch.

BM25 is rebuilt from the corpus held in memory — suitable for up to ~100k chunks.
For >100k: persist corpus to Redis/S3 and lazy-load.
"""
import re
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app.core.logging import get_logger

logger = get_logger(__name__)

_TOKENIZE_RE = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer — lowercases."""
    return _TOKENIZE_RE.findall(text.lower())


class BM25SparseRetriever:
    """
    In-memory BM25 retriever.
    - Documents are added incrementally via `add_documents`
    - Index is rebuilt lazily on first query after a change
    - Thread-safe for read-heavy workloads (queries >> writes)
    """

    def __init__(self) -> None:
        self._corpus: list[list[str]] = []      # tokenized documents
        self._chunk_ids: list[str] = []
        self._raw_texts: dict[str, str] = {}    # chunk_id → original text
        self._bm25: Optional[BM25Okapi] = None
        self._dirty: bool = False

    def add_documents(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Add documents to the corpus. Marks index as dirty."""
        for cid, text in zip(chunk_ids, texts):
            if cid not in self._raw_texts:
                tokens = _tokenize(text)
                self._corpus.append(tokens)
                self._chunk_ids.append(cid)
                self._raw_texts[cid] = text
        self._dirty = True
        logger.debug("bm25_docs_added", total=len(self._chunk_ids))

    def _build_index(self) -> None:
        if not self._corpus:
            return
        self._bm25 = BM25Okapi(self._corpus)
        self._dirty = False
        logger.info("bm25_index_built", corpus_size=len(self._corpus))

    def query(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """
        Returns list of (chunk_id, normalized_score) sorted by score desc.
        Scores normalized to [0, 1] range.
        """
        if not self._corpus:
            return []

        if self._dirty or self._bm25 is None:
            self._build_index()

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: np.ndarray = self._bm25.get_scores(query_tokens)  # type: ignore[union-attr]

        # Normalize to [0, 1]
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score
        else:
            return []

        # Get top-k indices
        top_indices = np.argpartition(scores, -min(top_k, len(scores)))[
            -min(top_k, len(scores)) :
        ]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            (self._chunk_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    @property
    def corpus_size(self) -> int:
        return len(self._chunk_ids)

    def remove_document(self, chunk_ids_to_remove: set[str]) -> None:
        """Remove chunks from the corpus. Triggers index rebuild."""
        keep_indices = [
            i for i, cid in enumerate(self._chunk_ids)
            if cid not in chunk_ids_to_remove
        ]
        self._corpus = [self._corpus[i] for i in keep_indices]
        self._chunk_ids = [self._chunk_ids[i] for i in keep_indices]
        for cid in chunk_ids_to_remove:
            self._raw_texts.pop(cid, None)
        self._dirty = True
