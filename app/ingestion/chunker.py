"""
Parent-Child Chunking Strategy.

Parent chunks = full semantic sections (larger context for LLM).
Child chunks = small semantic units used for embedding & retrieval.

When a child chunk is retrieved, we expand it back to its parent
to give the LLM fuller context — this is the core of multi-vector retrieval.
"""
import uuid
from typing import Optional

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging import get_logger
from app.core.models import ChunkType, DocumentChunk, DocumentMetadata, ParsedDocument

logger = get_logger(__name__)

# Token encoder for accurate token counting
_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _char_limit_from_tokens(token_limit: int, chars_per_token: float = 3.5) -> int:
    """Approximate character limit for a given token budget."""
    return int(token_limit * chars_per_token)


class ParentChildChunker:
    """
    Two-level chunking:
      Level 1 (Parent): Large chunks (~1500 tokens) capturing full sections.
      Level 2 (Child):  Small chunks (~400 tokens) derived from each parent.

    Child chunks store parent_id so we can expand context at query time.
    """

    def __init__(self) -> None:
        # Parent splitter — large windows for rich context
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=_char_limit_from_tokens(settings.PARENT_CHUNK_SIZE),
            chunk_overlap=_char_limit_from_tokens(settings.PARENT_CHUNK_OVERLAP),
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        # Child splitter — small, semantically dense units
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=_char_limit_from_tokens(settings.CHILD_CHUNK_SIZE),
            chunk_overlap=_char_limit_from_tokens(settings.CHILD_CHUNK_OVERLAP),
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(self, raw_text: str, metadata: DocumentMetadata) -> ParsedDocument:
        doc = ParsedDocument(
            doc_id=metadata.doc_id,
            raw_text=raw_text,
            metadata=metadata,
        )

        parent_texts = self._parent_splitter.split_text(raw_text)
        logger.info(
            "parent_chunks_created",
            doc_id=metadata.doc_id,
            count=len(parent_texts),
        )

        for p_idx, parent_text in enumerate(parent_texts):
            parent_id = str(uuid.uuid4())
            parent_chunk = DocumentChunk(
                chunk_id=parent_id,
                doc_id=metadata.doc_id,
                parent_id=None,
                chunk_type=ChunkType.PARENT,
                content=parent_text,
                token_count=_count_tokens(parent_text),
                char_count=len(parent_text),
                chunk_index=p_idx,
                metadata=metadata,
            )
            doc.parent_chunks.append(parent_chunk)

            # Derive child chunks from this parent
            child_texts = self._child_splitter.split_text(parent_text)
            for c_idx, child_text in enumerate(child_texts):
                if not child_text.strip():
                    continue
                child_chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=metadata.doc_id,
                    parent_id=parent_id,
                    chunk_type=ChunkType.CHILD,
                    content=child_text,
                    token_count=_count_tokens(child_text),
                    char_count=len(child_text),
                    chunk_index=c_idx,
                    metadata=metadata,
                )
                doc.child_chunks.append(child_chunk)

        logger.info(
            "child_chunks_created",
            doc_id=metadata.doc_id,
            count=len(doc.child_chunks),
        )
        return doc
