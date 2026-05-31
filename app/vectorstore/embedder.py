"""
Embedding service — wraps OpenAI embeddings with:
  - Redis caching (24-hour TTL)
  - Automatic batching (up to 100 texts per API call)
  - Retry logic with exponential backoff
  - Async throughout
"""
import asyncio
import hashlib
from typing import Optional

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.cache.redis_cache import RedisCache
from app.core.config import settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _embedding_cache_key(text: str, model: str) -> str:
    content_hash = hashlib.sha256(f"{model}:{text}".encode()).hexdigest()
    return f"emb:{content_hash}"


class EmbeddingService:
    """
    Async OpenAI embedding service with Redis-backed caching and batching.
    """

    def __init__(self, cache: RedisCache) -> None:
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.OPENAI_EMBEDDING_MODEL
        self._batch_size = settings.EMBEDDING_BATCH_SIZE
        self._cache = cache

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _call_openai(self, texts: list[str]) -> list[list[float]]:
        """Raw API call — wrapped in retry logic."""
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            logger.warning("openai_embedding_error", error=str(exc))
            raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single text, checking cache first."""
        cache_key = _embedding_cache_key(text, self._model)

        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        embeddings = await self._call_openai([text])
        embedding = embeddings[0]

        await self._cache.set(cache_key, embedding, ttl=settings.REDIS_EMBEDDING_TTL)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts efficiently.
        - Checks cache for each text first
        - Only calls OpenAI for cache misses
        - Reassembles results in original order
        """
        if not texts:
            return []

        results: list[Optional[list[float]]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache for all texts
        cache_tasks = [
            self._cache.get(_embedding_cache_key(t, self._model)) for t in texts
        ]
        cached_results = await asyncio.gather(*cache_tasks)

        for idx, cached in enumerate(cached_results):
            if cached is not None:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(texts[idx])

        logger.debug(
            "embedding_cache_stats",
            total=len(texts),
            cached=len(texts) - len(uncached_texts),
            uncached=len(uncached_texts),
        )

        if uncached_texts:
            # Process in batches to respect API limits
            all_new_embeddings: list[list[float]] = []
            for i in range(0, len(uncached_texts), self._batch_size):
                batch = uncached_texts[i : i + self._batch_size]
                batch_embeddings = await self._call_openai(batch)
                all_new_embeddings.extend(batch_embeddings)

            # Store in cache and fill results
            cache_set_tasks = []
            for idx, (orig_idx, embedding) in enumerate(
                zip(uncached_indices, all_new_embeddings)
            ):
                results[orig_idx] = embedding
                cache_key = _embedding_cache_key(uncached_texts[idx], self._model)
                cache_set_tasks.append(
                    self._cache.set(cache_key, embedding, ttl=settings.REDIS_EMBEDDING_TTL)
                )
            await asyncio.gather(*cache_set_tasks)

        # All slots should be filled now
        final: list[list[float]] = [r for r in results if r is not None]
        if len(final) != len(texts):
            raise EmbeddingError("Embedding count mismatch after batch processing")
        return final
