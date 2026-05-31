"""
Dependency injection — FastAPI lifespan-managed singletons.
All services initialized once at startup and torn down gracefully.
"""
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.cache.redis_cache import RedisCache
from app.core.logging import get_logger
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.sparse_retriever import BM25SparseRetriever
from app.services.rag_pipeline import RAGPipeline
from app.vectorstore.embedder import EmbeddingService
from app.vectorstore.pinecone_store import PineconeVectorStore

logger = get_logger(__name__)


class Container:
    """Service container holding all initialized singletons."""

    def __init__(self) -> None:
        self.cache: RedisCache = RedisCache()
        self.vector_store: PineconeVectorStore = PineconeVectorStore()
        self.embedder: EmbeddingService = EmbeddingService(cache=self.cache)
        self.sparse_retriever: BM25SparseRetriever = BM25SparseRetriever()
        self.retriever: HybridRetriever = HybridRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            sparse_retriever=self.sparse_retriever,
        )
        self.rag_pipeline: RAGPipeline = RAGPipeline(
            retriever=self.retriever,
            cache=self.cache,
        )
        self.ingestion_pipeline: IngestionPipeline = IngestionPipeline(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )

    async def startup(self) -> None:
        logger.info("container_startup_begin")
        await self.cache.initialize()
        await self.vector_store.initialize()
        logger.info("container_startup_complete")

    async def shutdown(self) -> None:
        logger.info("container_shutdown_begin")
        await self.cache.close()
        logger.info("container_shutdown_complete")


# Global singleton
_container: Container | None = None


def get_container() -> Container:
    if _container is None:
        raise RuntimeError("Container not initialized")
    return _container


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _container
    _container = Container()
    await _container.startup()
    yield
    await _container.shutdown()
    _container = None
