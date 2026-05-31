"""
Structured logging with structlog — ECS/CloudWatch compatible JSON output.
"""
import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from app.core.config import settings


def add_app_context(logger: Any, method: str, event_dict: EventDict) -> EventDict:
    event_dict["app"] = settings.APP_NAME
    event_dict["version"] = settings.APP_VERSION
    event_dict["environment"] = settings.ENVIRONMENT
    return event_dict


def setup_logging() -> None:
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_app_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.ENVIRONMENT == "production":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Propagate stdlib logs through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    for name in ("uvicorn", "uvicorn.access", "fastapi"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
