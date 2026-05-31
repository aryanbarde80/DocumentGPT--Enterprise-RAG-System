"""
Query endpoint — accepts a user question, runs the RAG pipeline, returns a grounded answer.
POST /api/v1/query
"""
from fastapi import APIRouter, HTTPException, status

from app.core.exceptions import LLMError, RetrievalError
from app.core.logging import get_logger
from app.core.models import QueryRequest, RAGResponse
from app.services.dependencies import get_container
from app.services.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/query", tags=["Query"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the document knowledge base",
    description=(
        "Submit a natural language question. The system performs hybrid retrieval "
        "(dense + sparse), expands context to parent chunks, and returns a grounded "
        "LLM-generated answer with source citations and a confidence score."
    ),
)
async def query_documents(request: QueryRequest) -> RAGResponse:
    """Run the full RAG pipeline for a user query."""

    logger.info(
        "query_request",
        query=request.query[:80],
        namespace=request.namespace,
        top_k=request.top_k,
    )

    try:
        container = get_container()
        pipeline: RAGPipeline = container.rag_pipeline
        response = await pipeline.run(request)
        return response

    except RetrievalError as exc:
        logger.error("query_retrieval_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Retrieval service error: {exc.message}",
        ) from exc
    except LLMError as exc:
        logger.error("query_llm_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {exc.message}",
        ) from exc
    except Exception as exc:
        logger.error("query_unexpected_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during query processing",
        ) from exc
