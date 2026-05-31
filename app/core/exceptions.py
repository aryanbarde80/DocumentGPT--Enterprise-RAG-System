"""
Domain-specific exceptions for clean error propagation.
"""
from typing import Optional


class DocumentGPTError(Exception):
    """Base exception."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class IngestionError(DocumentGPTError):
    """Raised when document ingestion fails."""


class UnsupportedFileTypeError(IngestionError):
    """Raised for unsupported file formats."""


class EmbeddingError(DocumentGPTError):
    """Raised when embedding generation fails."""


class VectorStoreError(DocumentGPTError):
    """Raised for Pinecone operation failures."""


class RetrievalError(DocumentGPTError):
    """Raised when retrieval fails."""


class LLMError(DocumentGPTError):
    """Raised when LLM call fails."""


class CacheError(DocumentGPTError):
    """Raised for Redis operation failures."""


class RateLimitError(DocumentGPTError):
    """Raised when rate limit is exceeded."""


class ValidationError(DocumentGPTError):
    """Raised for input validation failures."""
