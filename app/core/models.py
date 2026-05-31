"""
Domain models — the lingua franca of the entire system.
All inter-module communication uses these typed objects.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Enums ────────────────────────────────────────────────────────────────────

class FileType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    MARKDOWN = "md"


class ChunkType(str, Enum):
    PARENT = "parent"
    CHILD = "child"


class QueryStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


# ─── Document Models ──────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_file: str
    file_type: FileType
    file_size_bytes: int = 0
    page_count: Optional[int] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    namespace: str = "default"
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    parent_id: Optional[str] = None          # None for parent chunks
    chunk_type: ChunkType
    content: str
    token_count: int = 0
    char_count: int = 0
    chunk_index: int = 0
    metadata: DocumentMetadata
    embedding: Optional[list[float]] = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk content cannot be empty")
        return v.strip()


class ParsedDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_text: str
    metadata: DocumentMetadata
    parent_chunks: list[DocumentChunk] = Field(default_factory=list)
    child_chunks: list[DocumentChunk] = Field(default_factory=list)


# ─── Retrieval Models ─────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    parent_id: Optional[str]
    chunk_type: ChunkType
    content: str
    source_file: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    total_tokens: int = 0
    retrieval_latency_ms: float = 0.0


# ─── RAG Models ───────────────────────────────────────────────────────────────

class Source(BaseModel):
    doc_id: str
    chunk_id: str
    source_file: str
    relevance_score: float
    excerpt: str


class RAGResponse(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    answer: str
    sources: list[Source]
    confidence_score: float = Field(ge=0.0, le=1.0)
    status: QueryStatus = QueryStatus.SUCCESS
    cached: bool = False
    latency_ms: float = 0.0
    model_used: str = ""
    tokens_used: int = 0


# ─── API Request / Response Models ───────────────────────────────────────────

class IngestRequest(BaseModel):
    namespace: str = "default"
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    status: str
    file_name: str
    parent_chunks: int
    child_chunks: int
    ingestion_time_ms: float
    namespace: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    namespace: str = "default"
    top_k: int = Field(default=5, ge=1, le=20)
    rewrite_query: bool = True
    include_sources: bool = True

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return v.strip()


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    services: dict[str, str]
    uptime_seconds: float
