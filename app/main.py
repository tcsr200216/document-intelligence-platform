from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, Field
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from app.answering import AnswerProviderError, AnswerStatus, build_answer_generator
from app.cache import build_embedding_cache
from app.config import settings
from app.embeddings import EmbeddingProviderError, build_embedder
from app.observability import (
    ANSWER_OUTCOMES,
    DOCUMENT_DELETIONS,
    DOCUMENT_UPLOADS,
    EMBEDDING_CACHE_OPERATIONS,
    INGESTED_CHUNKS,
    INGESTION_DURATION,
    INGESTION_STAGES,
    RETRIEVAL_REQUESTS,
    RETRIEVAL_RESULTS,
    HttpObservabilityMiddleware,
    metrics_response,
)
from app.pagination import (
    InvalidDocumentCursor,
    decode_document_cursor,
    encode_document_cursor,
)
from app.parsing import DocumentParseError, chunk_document, parse_document
from app.repository import (
    DocumentRecord,
    DocumentRepository,
    DuplicateDocumentError,
    build_document_repository,
)
from app.retrieval import HybridRetriever
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
embedding_cache = build_embedding_cache(
    settings.redis_url,
    ttl_seconds=settings.embedding_cache_ttl_seconds,
    namespace=settings.embedding_cache_namespace,
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
    status: str


class DocumentListResponse(BaseModel):
    items: list[DocumentDetailResponse]
    next_cursor: str | None


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
app.add_middleware(HttpObservabilityMiddleware)


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
    if not embedding_cache.is_ready():
        raise HTTPException(status_code=503, detail="Embedding cache is unavailable.")
    return {
        "status": "ready",
        "embedding_provider": embedder.provider,
        "embedding_model": embedder.model_version,
        "embedding_dimensions": embedder.dimensions,
        "answer_provider": answer_generator.provider,
        "vector_store": vector_store.backend,
        "document_store": document_repository.backend,
        "embedding_cache": embedding_cache.backend,
    }


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {"service": "document-intelligence-platform", "status": "running"}


@app.get("/metrics", include_in_schema=False)
async def metrics():
    return metrics_response()


def _embed_query(text: str, operation: str) -> tuple[float, ...] | list[float]:
    text = text.strip()
    try:
        cached = embedding_cache.get(
            text,
            model_version=embedder.model_version,
            dimensions=embedder.dimensions,
        )
    except RedisError:
        EMBEDDING_CACHE_OPERATIONS.labels("get", "error").inc()
        cached = None
    else:
        if cached is not None:
            EMBEDDING_CACHE_OPERATIONS.labels("get", "hit").inc()
            return cached
        EMBEDDING_CACHE_OPERATIONS.labels("get", "miss").inc()

    try:
        vector = embedder.embed([text])[0]
    except EmbeddingProviderError as exc:
        RETRIEVAL_REQUESTS.labels(operation, "embedding_error").inc()
        if operation == "question":
            ANSWER_OUTCOMES.labels(answer_generator.provider, "provider_error").inc()
        raise HTTPException(status_code=503, detail="Embedding provider is unavailable.") from exc

    try:
        embedding_cache.set(
            text,
            vector,
            model_version=embedder.model_version,
            dimensions=embedder.dimensions,
        )
    except RedisError:
        EMBEDDING_CACHE_OPERATIONS.labels("set", "error").inc()
    else:
        EMBEDDING_CACHE_OPERATIONS.labels("set", "success").inc()
    return vector


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    tags=["documents"],
)
async def upload_document(
    response: Response,
    file: Annotated[UploadFile, File(...)],
) -> DocumentUploadResponse:
    """Parse, chunk, embed, index, and persist traceable document metadata."""
    filename = file.filename or "unnamed"
    extension = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower()
    document_format = extension.removeprefix(".") or "unknown"
    started = time.perf_counter()

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
    sha256 = hashlib.sha256(payload).hexdigest()

    try:
        existing = document_repository.get_by_sha256(sha256)
    except SQLAlchemyError as exc:
        DOCUMENT_UPLOADS.labels("lookup_error", document_format).inc()
        raise HTTPException(status_code=503, detail="Document repository is unavailable.") from exc
    if existing is not None:
        if existing.status != "indexed":
            DOCUMENT_UPLOADS.labels("in_progress", document_format).inc()
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Identical document ingestion is already in progress.",
                    "document_id": existing.document_id,
                },
            )
        response.status_code = 200
        DOCUMENT_UPLOADS.labels("deduplicated", document_format).inc()
        return _upload_response(existing, "already_indexed")

    try:
        parsed = parse_document(filename, payload)
    except DocumentParseError as exc:
        INGESTION_STAGES.labels("parse", "error", document_format).inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    INGESTION_STAGES.labels("parse", "success", document_format).inc()

    chunks = chunk_document(parsed)
    try:
        vectors = embedder.embed([chunk.text for chunk in chunks])
    except EmbeddingProviderError as exc:
        INGESTION_STAGES.labels("embed", "error", document_format).inc()
        raise HTTPException(status_code=503, detail="Embedding provider is unavailable.") from exc
    INGESTION_STAGES.labels("embed", "success", document_format).inc()
    document_id = str(uuid4())
    uploaded_at = datetime.now(UTC)

    # Persist the citation source before publishing it to the retriever. A
    # successful search result therefore always has durable metadata when SQL is configured.
    try:
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            content_type=content_type or "application/octet-stream",
            size_bytes=len(payload),
            sha256=sha256,
            uploaded_at=uploaded_at,
            character_count=parsed.character_count,
            chunks=tuple(chunks),
            status="indexing",
        )
        document_repository.save(record)
    except DuplicateDocumentError:
        existing = document_repository.get_by_sha256(sha256)
        if existing is not None and existing.status == "indexed":
            response.status_code = 200
            DOCUMENT_UPLOADS.labels("deduplicated", document_format).inc()
            return _upload_response(existing, "already_indexed")
        DOCUMENT_UPLOADS.labels("in_progress", document_format).inc()
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Identical document ingestion is already in progress.",
                "document_id": existing.document_id if existing else None,
            },
        )
    except SQLAlchemyError as exc:
        INGESTION_STAGES.labels("persist", "error", document_format).inc()
        raise HTTPException(status_code=503, detail="Document repository is unavailable.") from exc
    INGESTION_STAGES.labels("persist", "success", document_format).inc()
    try:
        vector_store.replace_document(document_id, chunks, vectors)
    except SQLAlchemyError as exc:
        INGESTION_STAGES.labels("index", "error", document_format).inc()
        try:
            document_repository.delete(document_id)
        except SQLAlchemyError:
            INGESTION_STAGES.labels("rollback", "error", document_format).inc()
        else:
            INGESTION_STAGES.labels("rollback", "success", document_format).inc()
        raise HTTPException(status_code=503, detail="Vector store is unavailable.") from exc
    INGESTION_STAGES.labels("index", "success", document_format).inc()
    try:
        document_repository.mark_indexed(document_id)
    except SQLAlchemyError as exc:
        INGESTION_STAGES.labels("publish", "error", document_format).inc()
        try:
            vector_store.delete_document(document_id)
            document_repository.delete(document_id)
        except SQLAlchemyError:
            INGESTION_STAGES.labels("rollback", "error", document_format).inc()
        else:
            INGESTION_STAGES.labels("rollback", "success", document_format).inc()
        raise HTTPException(status_code=503, detail="Document repository is unavailable.") from exc
    INGESTION_STAGES.labels("publish", "success", document_format).inc()
    INGESTION_STAGES.labels("completed", "success", document_format).inc()
    INGESTION_DURATION.labels(document_format).observe(time.perf_counter() - started)
    INGESTED_CHUNKS.labels(document_format).observe(len(chunks))

    DOCUMENT_UPLOADS.labels("indexed", document_format).inc()
    return _upload_response(
        DocumentRecord(
            document_id=document_id,
            filename=filename,
            content_type=content_type or "application/octet-stream",
            size_bytes=len(payload),
            sha256=sha256,
            uploaded_at=uploaded_at,
            character_count=parsed.character_count,
            chunks=tuple(chunks),
            status="indexed",
        ),
        "indexed",
    )


def _upload_response(record: DocumentRecord, status_value: str) -> DocumentUploadResponse:
    return DocumentUploadResponse(
        document_id=record.document_id,
        filename=record.filename,
        content_type=record.content_type,
        size_bytes=record.size_bytes,
        sha256=record.sha256,
        status=status_value,
        uploaded_at=record.uploaded_at,
        character_count=record.character_count,
        chunk_count=len(record.chunks),
    )


@app.get("/documents", response_model=DocumentListResponse, tags=["documents"])
async def list_documents(
    limit: int = Query(default=25, ge=1, le=100),
    cursor: str | None = Query(default=None, max_length=512),
    status_filter: Literal["indexing", "indexed"] | None = Query(default=None),
) -> DocumentListResponse:
    """List durable document metadata with stable keyset pagination."""
    try:
        decoded_cursor = (
            decode_document_cursor(cursor, status_filter) if cursor is not None else None
        )
    except InvalidDocumentCursor as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        page = document_repository.list_page(
            limit=limit,
            cursor=decoded_cursor,
            status=status_filter,
        )
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Document repository is unavailable.") from exc
    return DocumentListResponse(
        items=[
            DocumentDetailResponse(
                document_id=item.document_id,
                filename=item.filename,
                content_type=item.content_type,
                size_bytes=item.size_bytes,
                sha256=item.sha256,
                uploaded_at=item.uploaded_at,
                character_count=item.character_count,
                chunk_count=item.chunk_count,
                status=item.status,
            )
            for item in page.items
        ],
        next_cursor=(
            encode_document_cursor(page.next_cursor, status_filter)
            if page.next_cursor is not None
            else None
        ),
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
        status=record.status,
    )


@app.delete(
    "/documents/{document_id}",
    status_code=204,
    tags=["documents"],
)
async def delete_document(document_id: str) -> Response:
    """Remove retrieval vectors first, then their durable citation source."""
    try:
        record = document_repository.get(document_id)
    except SQLAlchemyError as exc:
        DOCUMENT_DELETIONS.labels("metadata_error").inc()
        raise HTTPException(status_code=503, detail="Document repository is unavailable.") from exc
    if record is None:
        DOCUMENT_DELETIONS.labels("not_found").inc()
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        vector_store.delete_document(document_id)
    except SQLAlchemyError as exc:
        DOCUMENT_DELETIONS.labels("vector_error").inc()
        raise HTTPException(status_code=503, detail="Vector store is unavailable.") from exc

    try:
        document_repository.delete(document_id)
    except SQLAlchemyError as exc:
        DOCUMENT_DELETIONS.labels("metadata_error").inc()
        raise HTTPException(
            status_code=503,
            detail="Document vectors were removed; retry metadata deletion.",
        ) from exc

    DOCUMENT_DELETIONS.labels("success").inc()
    return Response(status_code=204)


@app.post("/search", response_model=list[SearchResult], tags=["retrieval"])
async def search_documents(request: SearchRequest) -> list[SearchResult]:
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Search query must contain readable text.")

    query_vector = _embed_query(request.query, "search")
    try:
        hits = HybridRetriever(vector_store).retrieve(
            request.query,
            query_vector,
            limit=request.limit,
            document_id=request.document_id,
        )
    except SQLAlchemyError as exc:
        RETRIEVAL_REQUESTS.labels("search", "backend_error").inc()
        raise HTTPException(status_code=503, detail="Vector store is unavailable.") from exc
    RETRIEVAL_REQUESTS.labels("search", "success").inc()
    RETRIEVAL_RESULTS.labels("search").observe(len(hits))
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

    query_vector = _embed_query(request.question, "question")
    try:
        hits = HybridRetriever(vector_store).retrieve(
            request.question,
            query_vector,
            limit=request.context_limit,
            document_id=request.document_id,
        )
    except SQLAlchemyError as exc:
        RETRIEVAL_REQUESTS.labels("question", "backend_error").inc()
        ANSWER_OUTCOMES.labels(answer_generator.provider, "retrieval_error").inc()
        raise HTTPException(status_code=503, detail="Vector store is unavailable.") from exc
    RETRIEVAL_REQUESTS.labels("question", "success").inc()
    RETRIEVAL_RESULTS.labels("question").observe(len(hits))
    try:
        generated = answer_generator.generate(request.question, hits)
    except AnswerProviderError as exc:
        ANSWER_OUTCOMES.labels(answer_generator.provider, "provider_error").inc()
        raise HTTPException(status_code=503, detail="Answer provider is unavailable.") from exc
    ANSWER_OUTCOMES.labels(answer_generator.provider, generated.status).inc()

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
