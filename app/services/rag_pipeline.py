"""
RAG pipeline — the orchestration layer.

Flow:
  query
    → optional query rewriting (LLM)
    → query embedding (cached)
    → hybrid retrieval (dense + sparse)
    → parent expansion + deduplication
    → context compression
    → LLM answer generation (grounded only in context)
    → confidence scoring
    → structured JSON response

Caching: full pipeline output cached in Redis keyed by hash(query+namespace).
"""
import time
from textwrap import dedent
from typing import Optional

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.cache.redis_cache import RedisCache
from app.core.config import settings
from app.core.exceptions import LLMError, RetrievalError
from app.core.logging import get_logger
from app.core.models import QueryRequest, QueryStatus, RAGResponse, RetrievedChunk, Source
from app.retrieval.hybrid_retriever import HybridRetriever

logger = get_logger(__name__)


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = dedent("""
    You are a precise, factual question-answering assistant for an enterprise knowledge base.

    STRICT RULES:
    1. Answer ONLY from the provided context. Do not use outside knowledge.
    2. If the answer cannot be found in the context, say: "I cannot find this information in the provided documents."
    3. Always cite the source document(s) supporting your answer.
    4. Be concise but complete. Use bullet points for multi-part answers.
    5. Never speculate or extrapolate beyond what the context states.
    6. If context is contradictory, acknowledge the contradiction.
""").strip()

QUERY_REWRITE_PROMPT = dedent("""
    Rewrite the following user query to be more precise and search-friendly for a document retrieval system.
    Remove filler words, expand abbreviations, and add relevant synonyms.
    Return ONLY the rewritten query, nothing else.

    Original query: {query}
""").strip()

RAG_USER_PROMPT = dedent("""
    CONTEXT DOCUMENTS:
    {context}

    ---
    USER QUESTION: {question}

    Provide a comprehensive answer based ONLY on the context above.
    At the end, include a confidence level (0.0-1.0) on a line: CONFIDENCE: <float>
""").strip()


# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline with caching, query rewriting, and structured output.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        cache: RedisCache,
    ) -> None:
        self._retriever = retriever
        self._cache = cache
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _call_llm(
        self,
        messages: list[dict],
        max_tokens: int = settings.OPENAI_MAX_TOKENS,
    ) -> tuple[str, int]:
        """Call OpenAI chat completion. Returns (response_text, total_tokens)."""
        try:
            response = await self._client.chat.completions.create(
                model=settings.OPENAI_CHAT_MODEL,
                messages=messages,  # type: ignore[arg-type]
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            return content, tokens
        except Exception as exc:
            raise LLMError(f"OpenAI call failed: {exc}") from exc

    async def _rewrite_query(self, query: str) -> str:
        """Optional query rewriting for better retrieval coverage."""
        try:
            messages = [
                {"role": "user", "content": QUERY_REWRITE_PROMPT.format(query=query)},
            ]
            rewritten, _ = await self._call_llm(messages, max_tokens=200)
            rewritten = rewritten.strip()
            if rewritten and len(rewritten) > 5:
                logger.debug("query_rewritten", original=query[:60], rewritten=rewritten[:60])
                return rewritten
        except Exception as exc:
            logger.warning("query_rewrite_failed", error=str(exc))
        return query  # Fall back to original

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}: {chunk.source_file}]\n{chunk.content}"
            )
        return "\n\n---\n\n".join(parts)

    def _extract_confidence(self, llm_output: str) -> tuple[str, float]:
        """Parse CONFIDENCE: <float> from LLM output."""
        lines = llm_output.strip().splitlines()
        confidence = 0.7  # Default
        answer_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(stripped.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass
            else:
                answer_lines.append(line)

        answer = "\n".join(answer_lines).strip()
        return answer, confidence

    def _build_sources(self, chunks: list[RetrievedChunk]) -> list[Source]:
        """Convert retrieved chunks to Source objects."""
        return [
            Source(
                doc_id=c.doc_id,
                chunk_id=c.chunk_id,
                source_file=c.source_file,
                relevance_score=round(c.hybrid_score, 4),
                excerpt=c.content[:300] + ("..." if len(c.content) > 300 else ""),
            )
            for c in chunks
        ]

    async def run(self, request: QueryRequest) -> RAGResponse:
        """
        Execute the full RAG pipeline.
        Returns a structured RAGResponse grounded in retrieved context.
        """
        start_ts = time.monotonic()

        # 1. Check response cache
        cached_response = await self._cache.get_response(request.query, request.namespace)
        if cached_response:
            logger.info("cache_hit_response", query=request.query[:60])
            response = RAGResponse(**cached_response)
            response.cached = True
            response.latency_ms = round((time.monotonic() - start_ts) * 1000, 2)
            return response

        # 2. Optional query rewriting
        search_query = request.query
        if request.rewrite_query:
            search_query = await self._rewrite_query(request.query)

        # 3. Hybrid retrieval
        try:
            retrieval_result = await self._retriever.retrieve(
                query=search_query,
                top_k=request.top_k,
                namespace=request.namespace,
            )
        except Exception as exc:
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

        chunks = retrieval_result.chunks

        if not chunks:
            # No relevant context found
            answer = "I cannot find relevant information in the provided documents to answer this question."
            response = RAGResponse(
                query=request.query,
                answer=answer,
                sources=[],
                confidence_score=0.0,
                status=QueryStatus.PARTIAL,
                latency_ms=round((time.monotonic() - start_ts) * 1000, 2),
                model_used=settings.OPENAI_CHAT_MODEL,
                tokens_used=0,
            )
            return response

        # 4. Build context and call LLM
        context = self._build_context(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": RAG_USER_PROMPT.format(
                    context=context,
                    question=request.query,
                ),
            },
        ]

        llm_output, tokens_used = await self._call_llm(messages)

        # 5. Parse answer + confidence
        answer, confidence_score = self._extract_confidence(llm_output)

        # 6. Build sources
        sources = self._build_sources(chunks) if request.include_sources else []

        elapsed_ms = round((time.monotonic() - start_ts) * 1000, 2)

        response = RAGResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            confidence_score=confidence_score,
            status=QueryStatus.SUCCESS,
            cached=False,
            latency_ms=elapsed_ms,
            model_used=settings.OPENAI_CHAT_MODEL,
            tokens_used=tokens_used,
        )

        # 7. Cache the response
        await self._cache.set_response(
            request.query,
            request.namespace,
            response.model_dump(),
        )

        logger.info(
            "rag_pipeline_complete",
            query=request.query[:60],
            chunks_used=len(chunks),
            confidence=confidence_score,
            tokens=tokens_used,
            elapsed_ms=elapsed_ms,
        )

        return response
