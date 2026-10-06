from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    or_,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.parsing import TextChunk


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    uploaded_at: datetime
    character_count: int
    chunks: tuple[TextChunk, ...]
    status: str = "indexed"


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    uploaded_at: datetime
    character_count: int
    chunk_count: int
    status: str


@dataclass(frozen=True, slots=True)
class DocumentCursor:
    uploaded_at: datetime
    document_id: str


@dataclass(frozen=True, slots=True)
class DocumentPage:
    items: tuple[DocumentSummary, ...]
    next_cursor: DocumentCursor | None


class DuplicateDocumentError(ValueError):
    """Raised when another document already owns a content digest."""

    def __init__(self, sha256: str) -> None:
        super().__init__(f"A document with SHA-256 {sha256} already exists.")
        self.sha256 = sha256


class DocumentRepository(Protocol):
    @property
    def backend(self) -> str: ...

    def save(self, record: DocumentRecord) -> None:
        """Replace one document and all of its traceable chunks."""
        ...

    def get(self, document_id: str) -> DocumentRecord | None: ...

    def get_by_sha256(self, sha256: str) -> DocumentRecord | None: ...

    def list_page(
        self,
        limit: int,
        cursor: DocumentCursor | None = None,
        status: str | None = None,
    ) -> DocumentPage: ...

    def mark_indexed(self, document_id: str) -> None:
        """Publish a persisted document after its vectors are durable."""
        ...

    def delete(self, document_id: str) -> bool:
        """Delete document metadata and chunks, returning whether it existed."""
        ...

    def is_ready(self) -> bool: ...


class InMemoryDocumentRepository:
    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}

    @property
    def backend(self) -> str:
        return "memory"

    def save(self, record: DocumentRecord) -> None:
        duplicate = next(
            (
                document
                for document in self._documents.values()
                if document.sha256 == record.sha256 and document.document_id != record.document_id
            ),
            None,
        )
        if duplicate is not None:
            raise DuplicateDocumentError(record.sha256)
        self._documents[record.document_id] = record

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._documents.get(document_id)

    def get_by_sha256(self, sha256: str) -> DocumentRecord | None:
        return next(
            (document for document in self._documents.values() if document.sha256 == sha256),
            None,
        )

    def list_page(
        self,
        limit: int,
        cursor: DocumentCursor | None = None,
        status: str | None = None,
    ) -> DocumentPage:
        records = sorted(
            (
                record
                for record in self._documents.values()
                if status is None or record.status == status
            ),
            key=lambda record: (record.uploaded_at, record.document_id),
            reverse=True,
        )
        if cursor is not None:
            cursor_key = (cursor.uploaded_at, cursor.document_id)
            records = [
                record
                for record in records
                if (record.uploaded_at, record.document_id) < cursor_key
            ]
        window = records[: limit + 1]
        visible = window[:limit]
        items = tuple(_summary_from_record(record) for record in visible)
        next_cursor = (
            DocumentCursor(visible[-1].uploaded_at, visible[-1].document_id)
            if len(window) > limit
            else None
        )
        return DocumentPage(items=items, next_cursor=next_cursor)

    def mark_indexed(self, document_id: str) -> None:
        record = self._documents.get(document_id)
        if record is None:
            raise SQLAlchemyError("Document disappeared before indexing completed.")
        self._documents[document_id] = DocumentRecord(
            document_id=record.document_id,
            filename=record.filename,
            content_type=record.content_type,
            size_bytes=record.size_bytes,
            sha256=record.sha256,
            uploaded_at=record.uploaded_at,
            character_count=record.character_count,
            chunks=record.chunks,
            status="indexed",
        )

    def delete(self, document_id: str) -> bool:
        return self._documents.pop(document_id, None) is not None

    def is_ready(self) -> bool:
        return True


metadata = MetaData()
documents_table = Table(
    "documents",
    metadata,
    Column("document_id", String(36), primary_key=True),
    Column("filename", String(512), nullable=False),
    Column("content_type", String(128), nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("uploaded_at", DateTime(timezone=True), nullable=False),
    Column("character_count", Integer, nullable=False),
    Column("status", String(32), nullable=False, server_default="indexed"),
)
Index("uq_documents_sha256", documents_table.c.sha256, unique=True)
chunks_table = Table(
    "document_chunks",
    metadata,
    Column(
        "document_id",
        String(36),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("chunk_index", Integer, primary_key=True),
    Column("text", Text, nullable=False),
    Column("start_char", Integer, nullable=False),
    Column("end_char", Integer, nullable=False),
    Column("page_start", Integer, nullable=True),
    Column("page_end", Integer, nullable=True),
)


class SqlDocumentRepository:
    """SQLAlchemy adapter for durable document metadata and citation spans."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @property
    def backend(self) -> str:
        return "sql"

    @classmethod
    def from_url(cls, database_url: str) -> SqlDocumentRepository:
        return cls(create_engine(database_url, pool_pre_ping=True))

    def create_schema(self) -> None:
        metadata.create_all(self._engine)
        if self._engine.dialect.name == "postgresql":
            with self._engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_start INTEGER")
                )
                connection.execute(
                    text(
                        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS status "
                        "VARCHAR(32) NOT NULL DEFAULT 'indexed'"
                    )
                )
                connection.execute(
                    text(
                        "CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_sha256 "
                        "ON documents (sha256)"
                    )
                )
                connection.execute(
                    text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_end INTEGER")
                )

    def save(self, record: DocumentRecord) -> None:
        # The metadata row and all citation spans change in one transaction.
        try:
            with self._engine.begin() as connection:
                connection.execute(
                    delete(chunks_table).where(chunks_table.c.document_id == record.document_id)
                )
                connection.execute(
                    delete(documents_table).where(
                        documents_table.c.document_id == record.document_id
                    )
                )
                connection.execute(
                    insert(documents_table).values(
                        document_id=record.document_id,
                        filename=record.filename,
                        content_type=record.content_type,
                        size_bytes=record.size_bytes,
                        sha256=record.sha256,
                        uploaded_at=record.uploaded_at,
                        character_count=record.character_count,
                        status=record.status,
                    )
                )
                if record.chunks:
                    connection.execute(
                        insert(chunks_table),
                        [
                            {
                                "document_id": record.document_id,
                                "chunk_index": chunk.index,
                                "text": chunk.text,
                                "start_char": chunk.start_char,
                                "end_char": chunk.end_char,
                                "page_start": chunk.page_start,
                                "page_end": chunk.page_end,
                            }
                            for chunk in record.chunks
                        ],
                    )
        except IntegrityError as exc:
            raise DuplicateDocumentError(record.sha256) from exc

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._get_one(documents_table.c.document_id == document_id)

    def get_by_sha256(self, sha256: str) -> DocumentRecord | None:
        return self._get_one(documents_table.c.sha256 == sha256)

    def list_page(
        self,
        limit: int,
        cursor: DocumentCursor | None = None,
        status: str | None = None,
    ) -> DocumentPage:
        statement = select(documents_table).order_by(
            documents_table.c.uploaded_at.desc(),
            documents_table.c.document_id.desc(),
        )
        if status is not None:
            statement = statement.where(documents_table.c.status == status)
        if cursor is not None:
            cursor_time = cursor.uploaded_at
            if self._engine.dialect.name == "sqlite":
                cursor_time = cursor_time.replace(tzinfo=None)
            statement = statement.where(
                or_(
                    documents_table.c.uploaded_at < cursor_time,
                    (
                        (documents_table.c.uploaded_at == cursor_time)
                        & (documents_table.c.document_id < cursor.document_id)
                    ),
                )
            )
        with self._engine.connect() as connection:
            rows = connection.execute(statement.limit(limit + 1)).mappings().all()
            visible = rows[:limit]
            document_ids = [row["document_id"] for row in visible]
            chunk_counts = {}
            if document_ids:
                chunk_counts = dict(
                    connection.execute(
                        select(
                            chunks_table.c.document_id,
                            func.count(chunks_table.c.chunk_index),
                        )
                        .where(chunks_table.c.document_id.in_(document_ids))
                        .group_by(chunks_table.c.document_id)
                    ).all()
                )
        items = tuple(
            DocumentSummary(
                document_id=row["document_id"],
                filename=row["filename"],
                content_type=row["content_type"],
                size_bytes=row["size_bytes"],
                sha256=row["sha256"],
                uploaded_at=_as_utc(row["uploaded_at"]),
                character_count=row["character_count"],
                chunk_count=chunk_counts.get(row["document_id"], 0),
                status=row["status"],
            )
            for row in visible
        )
        next_cursor = (
            DocumentCursor(items[-1].uploaded_at, items[-1].document_id)
            if len(rows) > limit
            else None
        )
        return DocumentPage(items=items, next_cursor=next_cursor)

    def _get_one(self, predicate) -> DocumentRecord | None:
        with self._engine.connect() as connection:
            document = (
                connection.execute(select(documents_table).where(predicate))
                .mappings()
                .one_or_none()
            )
            if document is None:
                return None
            rows = (
                connection.execute(
                    select(chunks_table)
                    .where(chunks_table.c.document_id == document["document_id"])
                    .order_by(chunks_table.c.chunk_index)
                )
                .mappings()
                .all()
            )

        return DocumentRecord(
            document_id=document["document_id"],
            filename=document["filename"],
            content_type=document["content_type"],
            size_bytes=document["size_bytes"],
            sha256=document["sha256"],
            uploaded_at=document["uploaded_at"],
            character_count=document["character_count"],
            chunks=tuple(
                TextChunk(
                    index=row["chunk_index"],
                    text=row["text"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                )
                for row in rows
            ),
            status=document["status"],
        )

    def mark_indexed(self, document_id: str) -> None:
        with self._engine.begin() as connection:
            result = connection.execute(
                documents_table.update()
                .where(documents_table.c.document_id == document_id)
                .values(status="indexed")
            )
        if not result.rowcount:
            raise SQLAlchemyError("Document disappeared before indexing completed.")

    def delete(self, document_id: str) -> bool:
        with self._engine.begin() as connection:
            connection.execute(
                delete(chunks_table).where(chunks_table.c.document_id == document_id)
            )
            result = connection.execute(
                delete(documents_table).where(documents_table.c.document_id == document_id)
            )
        return bool(result.rowcount)

    def is_ready(self) -> bool:
        try:
            with self._engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False


def build_document_repository(database_url: str | None) -> DocumentRepository:
    if not database_url:
        return InMemoryDocumentRepository()
    repository = SqlDocumentRepository.from_url(database_url)
    repository.create_schema()
    return repository


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _summary_from_record(record: DocumentRecord) -> DocumentSummary:
    return DocumentSummary(
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
