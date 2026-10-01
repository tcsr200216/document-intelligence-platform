from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.embeddings import HashingEmbedder
from app.parsing import DocumentParseError, chunk_text, parse_text_document
from app.vector_store import InMemoryVectorStore

MAX_DOCUMENT_SIZE_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
}

embedder = HashingEmbedder()
vector_store = InMemoryVectorStore(dimensions=embedder.dimensions)


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    status: str
    uploaded_at: datetime
    character_count: int
    chunk_count: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    document_id: str | None = Field(default=None, min_length=1, max_length=128)
    limit: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    document_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    score: float


app = FastAPI(
    title="Document Intelligence Platform",
    version="0.1.0",
    description="Production-oriented document ingestion and retrieval API.",
)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Return a lightweight liveness response for local and container health checks."""
    return {"status": "ok"}


@app.get("/ready", tags=["system"])
async def ready() -> dict[str, str | int]:
    """Report whether the local embedding and retrieval components agree on dimensions."""
    if vector_store.dimensions != embedder.dimensions:
        raise HTTPException(status_code=503, detail="Embedding and vector-store dimensions differ.")
    return {
        "status": "ready",
        "embedding_dimensions": embedder.dimensions,
        "vector_store": "memory",
    }


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
    """Validate and synchronously index a UTF-8 text or Markdown document."""
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
        raise HTTPException(status_code=415, detail=f"Unsupported content type '{content_type}'.")

    if extension == ".pdf":
        raise HTTPException(
            status_code=501,
            detail="PDF upload is recognized but PDF text extraction is not implemented yet.",
        )

    payload = await file.read(MAX_DOCUMENT_SIZE_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded document is empty.")
    if len(payload) > MAX_DOCUMENT_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Document exceeds the 10 MB upload limit.")

    try:
        parsed = parse_text_document(filename, payload)
    except DocumentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    chunks = chunk_text(parsed.text)
    vectors = embedder.embed([chunk.text for chunk in chunks])
    document_id = str(uuid4())
    vector_store.replace_document(document_id, chunks, vectors)

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        content_type=content_type or "application/octet-stream",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        status="indexed",
        uploaded_at=datetime.now(UTC),
        character_count=parsed.character_count,
        chunk_count=len(chunks),
    )


@app.post(
    "/search",
    response_model=list[SearchResult],
    tags=["retrieval"],
)
async def search_documents(request: SearchRequest) -> list[SearchResult]:
    """Embed a query and return traceable source chunks ranked by cosine similarity."""
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Search query must contain readable text.")

    query_vector = embedder.embed([request.query])[0]
    hits = vector_store.search(
        query_vector,
        limit=request.limit,
        document_id=request.document_id,
    )
    return [
        SearchResult(
            document_id=hit.document_id,
            chunk_index=hit.chunk.index,
            text=hit.chunk.text,
            start_char=hit.chunk.start_char,
            end_char=hit.chunk.end_char,
            score=hit.score,
        )
        for hit in hits
    ]
