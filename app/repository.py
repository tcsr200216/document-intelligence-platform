from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

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


class DocumentRepository(Protocol):
    @property
    def backend(self) -> str:
        ...

    def save(self, record: DocumentRecord) -> None:
        """Replace one document and all of its traceable chunks."""
        ...

    def get(self, document_id: str) -> DocumentRecord | None:
        ...

    def delete(self, document_id: str) -> bool:
        """Delete document metadata and chunks, returning whether it existed."""
        ...

    def is_ready(self) -> bool:
        ...


class InMemoryDocumentRepository:
    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}

    @property
    def backend(self) -> str:
        return "memory"

    def save(self, record: DocumentRecord) -> None:
        self._documents[record.document_id] = record

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._documents.get(document_id)

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
    Column("sha256", String(64), nullable=False, index=True),
    Column("uploaded_at", DateTime(timezone=True), nullable=False),
    Column("character_count", Integer, nullable=False),
)
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
                    text(
                        "ALTER TABLE document_chunks "
                        "ADD COLUMN IF NOT EXISTS page_start INTEGER"
                    )
                )
                connection.execute(
                    text(
                        "ALTER TABLE document_chunks "
                        "ADD COLUMN IF NOT EXISTS page_end INTEGER"
                    )
                )

    def save(self, record: DocumentRecord) -> None:
        # The metadata row and all citation spans change in one transaction.
        with self._engine.begin() as connection:
            connection.execute(
                delete(chunks_table).where(
                    chunks_table.c.document_id == record.document_id
                )
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

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._engine.connect() as connection:
            document = connection.execute(
                select(documents_table).where(
                    documents_table.c.document_id == document_id
                )
            ).mappings().one_or_none()
            if document is None:
                return None
            rows = connection.execute(
                select(chunks_table)
                .where(chunks_table.c.document_id == document_id)
                .order_by(chunks_table.c.chunk_index)
            ).mappings().all()

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
        )

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
