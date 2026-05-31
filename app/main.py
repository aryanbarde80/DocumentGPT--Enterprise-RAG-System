"""
FastAPI application entry point.
Production-grade: async, structured logging, Prometheus metrics,
CORS, exception handlers, lifespan management.
"""
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.v1 import health, ingest, query
from app.core.config import settings
from app.core.exceptions import DocumentGPTError
from app.core.logging import get_logger, setup_logging
from app.services.dependencies import lifespan

setup_logging()
logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="DocumentGPT – Enterprise RAG System",
        description=(
            "Production-grade Retrieval-Augmented Generation system "
            "with hybrid dense+sparse retrieval, parent-child chunking, "
            "Redis caching, and OpenAI GPT-4o."
        ),
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
    )

    # ─── Middleware ───────────────────────────────────────────────────────────

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_id_and_logging(request: Request, call_next):
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        start = time.monotonic()
        response = await call_next(request)
        elapsed = round((time.monotonic() - start) * 1000, 2)
        logger.info(
            "http_request",
            status_code=response.status_code,
            elapsed_ms=elapsed,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    # ─── Exception Handlers ───────────────────────────────────────────────────

    @app.exception_handler(DocumentGPTError)
    async def domain_exception_handler(request: Request, exc: DocumentGPTError):
        logger.error("domain_error", error=exc.message, details=exc.details)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": exc.message, "details": exc.details},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "An unexpected error occurred"},
        )

    # ─── Prometheus Metrics ───────────────────────────────────────────────────
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics")

    # ─── Routers ──────────────────────────────────────────────────────────────
    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix)
    app.include_router(ingest.router, prefix=prefix)
    app.include_router(query.router, prefix=prefix)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
        }

    return app


app = create_app()
