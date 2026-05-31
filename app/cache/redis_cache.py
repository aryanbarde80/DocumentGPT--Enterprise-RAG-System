"""
Redis cache layer — async, connection-pooled, JSON serialization.
Handles both embedding vectors and full RAG responses.
"""
import hashlib
import json
from typing import Any, Optional

import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _query_cache_key(query: str, namespace: str) -> str:
    content = f"{namespace}:{query.lower().strip()}"
    h = hashlib.xxh64(content.encode()).hexdigest()
    return f"cache:{h}"


class RedisCache:
    """
    Async Redis client with connection pooling.
    Serializes/deserializes arbitrary JSON-compatible objects.
    """

    def __init__(self) -> None:
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None

    async def initialize(self) -> None:
        self._pool = ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            decode_responses=False,
        )
        self._client = aioredis.Redis(connection_pool=self._pool)
        await self._client.ping()
        logger.info("redis_connected", url=settings.REDIS_URL)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
        if self._pool:
            await self._pool.aclose()

    def _client_or_raise(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("Redis not initialized — call initialize() first")
        return self._client

    async def get(self, key: str) -> Optional[Any]:
        try:
            raw = await self._client_or_raise().get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("redis_get_error", key=key, error=str(exc))
            return None

    async def set(self, key: str, value: Any, ttl: int = settings.REDIS_TTL_SECONDS) -> bool:
        try:
            serialized = json.dumps(value, default=str)
            await self._client_or_raise().setex(key, ttl, serialized)
            return True
        except Exception as exc:
            logger.warning("redis_set_error", key=key, error=str(exc))
            return False

    async def delete(self, key: str) -> bool:
        try:
            result = await self._client_or_raise().delete(key)
            return result > 0
        except Exception as exc:
            logger.warning("redis_delete_error", key=key, error=str(exc))
            return False

    async def exists(self, key: str) -> bool:
        try:
            return bool(await self._client_or_raise().exists(key))
        except Exception:
            return False

    async def health_check(self) -> bool:
        try:
            return await self._client_or_raise().ping()
        except Exception:
            return False

    # ─── Convenience helpers for the RAG layer ────────────────────────────────

    async def get_response(self, query: str, namespace: str) -> Optional[dict]:
        key = _query_cache_key(query, namespace)
        result = await self.get(key)
        if result:
            logger.debug("cache_hit", query=query[:50])
        return result

    async def set_response(
        self, query: str, namespace: str, response: dict
    ) -> None:
        key = _query_cache_key(query, namespace)
        await self.set(key, response, ttl=settings.REDIS_TTL_SECONDS)
        logger.debug("cache_set", query=query[:50])
