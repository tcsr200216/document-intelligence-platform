from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.answering import AnswerProviderError, AnswerStatus, build_answer_generator
from app.config import settings
from app.embeddings import build_embedder
from app.parsing import DocumentParseError, chunk_document, parse_document
from app.repository import DocumentRecord, DocumentRepository, build_document_repository
from app.vector_store import build_vector_store

MAX_DOCUMENT_SIZE_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
SUPPORTED_CONTENT_TYPES = {"application/pdf", "text/plain", "text/markdown"}

embedder = build_embedder(
    provider=settings.embedding_provider,
    dimensions=settings.embedding_dimensions,
    openai_api_key=(
        settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    ),
    openai_model=settings.openai_embedding_model,
    openai_base_url=settings.openai_base_url,
    timeout_seconds=settings.embedding_timeout_seconds,
)
vector_store = build_vector_store(
    settings.database_url,
    dimensions=embedder.dimensions,
    model_version=embedder.model_version,
)
document_repository: DocumentRepository = build_document_repository(settings.database_url)
answer_generator = build_answer_generator(
    provider=settings.answer_provider,
    openai_api_key=(
        settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    ),
    openai_model=settings.openai_answer_model,
    openai_base_url=settings.openai_base_url,
    timeout_seconds=settings.answer_timeout_seconds,
    min_relevance=settings.answer_min_relevance,
)


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


class DocumentDetailResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
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
    page_start: int | None
    page_end: int | None
    score: float


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    document_id: str | None = Field(default=None, min_length=1, max_length=128)
    context_limit: int = Field(default=5, ge=1, le=10)


class CitationResponse(BaseModel):
    document_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    page_start: int | None
    page_end: int | None
    score: float


class QuestionResponse(BaseModel):
    answer: str
    status: AnswerStatus
    answer_provider: str
    citations: list[CitationResponse]


app = FastAPI(
    title="Document Intelligence Platform",
    version="0.1.0",
    description="Production-oriented document ingestion and retrieval API.",
)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["system"])
async def ready() -> dict[str, str | int]:
    if vector_store.dimensions != embedder.dimensions:
        raise HTTPException(status_code=503, detail="Embedding and vector-store dimensions differ.")
    if not document_repository.is_ready():
        raise HTTPException(status_code=503, detail="Document repository is unavailable.")
    if not vector_store.is_ready():
        raise HTTPException(status_code=503, detail="Vector store is unavailable.")
    return {
        "status": "ready",
        "embedding_provider": embedder.provider,
        "embedding_model": embedder.model_version,
        "embedding_dimensions": embedder.dimensions,
        "answer_provider": answer_generator.provider,
        "vector_store": vector_store.backend,
        "document_store": document_repository.backend,
    }


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {"service": "document-intelligence-platform", "status": "running"}


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    tags=["documents"],
)
async def upload_document(file: Annotated[UploadFile, File(...)]) -> DocumentUploadResponse:
    """Parse, chunk, embed, index, and persist traceable document metadata."""
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
    payload = await file.read(MAX_DOCUMENT_SIZE_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded document is empty.")
    if len(payload) > MAX_DOCUMENT_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Document exceeds the 10 MB upload limit.")

    try:
        parsed = parse_document(filename, payload)
    except DocumentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    chunks = chunk_document(parsed)
    vectors = embedder.embed([chunk.text for chunk in chunks])
    document_id = str(uuid4())
    uploaded_at = datetime.now(UTC)
    sha256 = hashlib.sha256(payload).hexdigest()

    # Persist the citation source before publishing it to the retriever. A
    # successful search result therefore always has durable metadata when SQL is configured.
    document_repository.save(
        DocumentRecord(
            document_id=document_id,
            filename=filename,
            content_type=content_type or "application/octet-stream",
            size_bytes=len(payload),
            sha256=sha256,
            uploaded_at=uploaded_at,
            character_count=parsed.character_count,
            chunks=tuple(chunks),
        )
    )
    vector_store.replace_document(document_id, chunks, vectors)

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        content_type=content_type or "application/octet-stream",
        size_bytes=len(payload),
        sha256=sha256,
        status="indexed",
        uploaded_at=uploaded_at,
        character_count=parsed.character_count,
        chunk_count=len(chunks),
    )


@app.get(
    "/documents/{document_id}",
    response_model=DocumentDetailResponse,
    tags=["documents"],
)
async def get_document(document_id: str) -> DocumentDetailResponse:
    record = document_repository.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentDetailResponse(
        document_id=record.document_id,
        filename=record.filename,
        content_type=record.content_type,
        size_bytes=record.size_bytes,
        sha256=record.sha256,
        uploaded_at=record.uploaded_at,
        character_count=record.character_count,
        chunk_count=len(record.chunks),
    )


@app.post("/search", response_model=list[SearchResult], tags=["retrieval"])
async def search_documents(request: SearchRequest) -> list[SearchResult]:
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Search query must contain readable text.")

    query_vector = embedder.embed([request.query])[0]
    hits = vector_store.search(
        query_vector, limit=request.limit, document_id=request.document_id
    )
    return [
        SearchResult(
            document_id=hit.document_id,
            chunk_index=hit.chunk.index,
            text=hit.chunk.text,
            start_char=hit.chunk.start_char,
            end_char=hit.chunk.end_char,
            page_start=hit.chunk.page_start,
            page_end=hit.chunk.page_end,
            score=hit.score,
        )
        for hit in hits
    ]


@app.post("/questions", response_model=QuestionResponse, tags=["question-answering"])
async def answer_question(request: QuestionRequest) -> QuestionResponse:
    """Retrieve source spans, answer from them, and return validated citations."""
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question must contain readable text.")

    query_vector = embedder.embed([request.question])[0]
    hits = vector_store.search(
        query_vector,
        limit=request.context_limit,
        document_id=request.document_id,
    )
    try:
        generated = answer_generator.generate(request.question, hits)
    except AnswerProviderError as exc:
        raise HTTPException(status_code=503, detail="Answer provider is unavailable.") from exc

    citations = [
        CitationResponse(
            document_id=hit.document_id,
            chunk_index=hit.chunk.index,
            text=hit.chunk.text,
            start_char=hit.chunk.start_char,
            end_char=hit.chunk.end_char,
            page_start=hit.chunk.page_start,
            page_end=hit.chunk.page_end,
            score=hit.score,
        )
        for index, hit in enumerate(hits)
        if index in generated.citation_indices
    ]
    return QuestionResponse(
        answer=generated.answer,
        status=generated.status,
        answer_provider=answer_generator.provider,
        citations=citations,
    )
