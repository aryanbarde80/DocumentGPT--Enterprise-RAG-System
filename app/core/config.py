"""
Core configuration module using pydantic-settings.
All secrets loaded from environment variables for ECS/12-factor compliance.
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ─── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "DocumentGPT"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ─── API Keys ──────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = "us-east-1"

    # ─── OpenAI ────────────────────────────────────────────────────────────────
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-4o"
    OPENAI_MAX_TOKENS: int = 2048
    OPENAI_TEMPERATURE: float = 0.0
    EMBEDDING_DIMENSIONS: int = 1536
    EMBEDDING_BATCH_SIZE: int = 100

    # ─── Pinecone ──────────────────────────────────────────────────────────────
    PINECONE_INDEX_NAME: str = "documentgpt-index"
    PINECONE_NAMESPACE: str = "default"
    PINECONE_METRIC: str = "cosine"
    PINECONE_TOP_K: int = 20

    # ─── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://redis:6379/0"
    REDIS_TTL_SECONDS: int = 3600          # 1 hour for responses
    REDIS_EMBEDDING_TTL: int = 86400       # 24 hours for embeddings
    REDIS_MAX_CONNECTIONS: int = 50

    # ─── Chunking ──────────────────────────────────────────────────────────────
    PARENT_CHUNK_SIZE: int = 1500          # tokens
    PARENT_CHUNK_OVERLAP: int = 150
    CHILD_CHUNK_SIZE: int = 400
    CHILD_CHUNK_OVERLAP: int = 50

    # ─── Retrieval ─────────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 5
    HYBRID_DENSE_WEIGHT: float = 0.7
    HYBRID_SPARSE_WEIGHT: float = 0.3
    MAX_CONTEXT_TOKENS: int = 6000
    RERANK_ENABLED: bool = True

    # ─── Concurrency ───────────────────────────────────────────────────────────
    MAX_CONCURRENT_INGEST: int = 10
    MAX_CONCURRENT_QUERIES: int = 200

    # ─── Security ──────────────────────────────────────────────────────────────
    API_KEY_HEADER: str = "X-API-Key"
    INTERNAL_API_KEY: Optional[str] = None

    # ─── AWS ECS ───────────────────────────────────────────────────────────────
    AWS_REGION: str = "us-east-1"
    ECS_TASK_DEFINITION: Optional[str] = None


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
