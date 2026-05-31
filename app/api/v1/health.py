"""
Health check endpoint — GET /api/v1/health
Returns status of all dependent services for load-balancer and ECS health probes.
"""
import time

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import get_logger
from app.core.models import HealthResponse
from app.services.dependencies import get_container

router = APIRouter(prefix="/health", tags=["Health"])
logger = get_logger(__name__)

_START_TIME = time.monotonic()


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System health check",
    description="Returns health status of all backend services (Redis, Pinecone).",
)
async def health_check() -> HealthResponse:
    """Aggregate health check for ECS/ALB target group health probes."""
    services: dict[str, str] = {}
    overall_healthy = True

    try:
        container = get_container()

        # Redis
        redis_ok = await container.cache.health_check()
        services["redis"] = "healthy" if redis_ok else "unhealthy"
        if not redis_ok:
            overall_healthy = False

        # Pinecone
        pinecone_ok = await container.vector_store.health_check()
        services["pinecone"] = "healthy" if pinecone_ok else "unhealthy"
        if not pinecone_ok:
            overall_healthy = False

        # BM25 corpus
        sparse_size = container.sparse_retriever.corpus_size
        services["bm25_corpus"] = f"loaded ({sparse_size} chunks)"

    except Exception as exc:
        logger.error("health_check_error", error=str(exc))
        overall_healthy = False
        services["error"] = str(exc)

    response_data = HealthResponse(
        status="healthy" if overall_healthy else "degraded",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        services=services,
        uptime_seconds=round(time.monotonic() - _START_TIME, 2),
    )

    http_status = status.HTTP_200_OK if overall_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=response_data.model_dump(), status_code=http_status)
