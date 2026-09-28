from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

MAX_DOCUMENT_SIZE_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
}


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    status: str
    uploaded_at: datetime


app = FastAPI(
    title="Document Intelligence Platform",
    version="0.1.0",
    description="Production-oriented document ingestion and retrieval API.",
)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Return a lightweight liveness response for local and container health checks."""
    return {"status": "ok"}


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {
        "service": "document-intelligence-platform",
        "status": "running",
    }


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    tags=["documents"],
)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Validate and accept a document for downstream ingestion.

    This endpoint intentionally stops at the ingestion boundary: persistence,
    text extraction, chunking, and embedding generation are separate pipeline
    stages that will be added incrementally.
    """
    filename = file.filename or "unnamed"
    extension = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file extension '{extension or 'none'}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )

    if content_type and content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{content_type}'.",
        )

    payload = await file.read(MAX_DOCUMENT_SIZE_BYTES + 1)

    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded document is empty.")

    if len(payload) > MAX_DOCUMENT_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Document exceeds the 10 MB upload limit.",
        )

    digest = hashlib.sha256(payload).hexdigest()

    return DocumentUploadResponse(
        document_id=str(uuid4()),
        filename=filename,
        content_type=content_type or "application/octet-stream",
        size_bytes=len(payload),
        sha256=digest,
        status="accepted",
        uploaded_at=datetime.now(UTC),
    )
