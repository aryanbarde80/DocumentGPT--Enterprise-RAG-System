"""
Document ingestion endpoint.
POST /api/v1/ingest — upload and process a document file.
"""
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.exceptions import IngestionError, UnsupportedFileTypeError
from app.core.logging import get_logger
from app.core.models import IngestResponse
from app.ingestion.pipeline import IngestionPipeline
from app.services.dependencies import get_container

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = get_logger(__name__)

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/markdown",
    "text/x-markdown",
    "application/octet-stream",  # Some clients send this for unknown types
}

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document",
    description=(
        "Upload a document (PDF, TXT, DOCX, or Markdown) for processing. "
        "The file is parsed, chunked with parent-child strategy, embedded, "
        "and stored in Pinecone."
    ),
)
async def ingest_document(
    file: UploadFile = File(..., description="Document file to ingest"),
    namespace: str = Form(default="default", description="Pinecone namespace for isolation"),
) -> IngestResponse:
    """Ingest a single document into the RAG system."""

    # Validate file provided
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )

    logger.info(
        "ingest_request",
        filename=file.filename,
        content_type=file.content_type,
        namespace=namespace,
    )

    # Read file content with size limit
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {MAX_FILE_SIZE_MB}MB limit",
        )
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    # Write to temp file for parser access
    suffix = Path(file.filename).suffix.lower() or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        container = get_container()
        pipeline: IngestionPipeline = container.ingestion_pipeline

        response = await pipeline.ingest_file(
            file_path=tmp_path,
            file_name=file.filename,
            namespace=namespace,
        )
        return response

    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except IngestionError as exc:
        logger.error("ingest_endpoint_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("ingest_unexpected_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during ingestion",
        ) from exc
    finally:
        # Always clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()
