from datetime import UTC, datetime

from sqlalchemy import create_engine

from app.parsing import TextChunk
from app.repository import (
    DocumentRecord,
    InMemoryDocumentRepository,
    SqlDocumentRepository,
)


def record(document_id: str = "doc-1") -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        filename="architecture.txt",
        content_type="text/plain",
        size_bytes=11,
        sha256="a" * 64,
        uploaded_at=datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
        character_count=11,
        chunks=(
            TextChunk(index=0, text="hello world", start_char=0, end_char=11),
        ),
    )


def test_memory_repository_round_trips_exact_chunk_offsets() -> None:
    repository = InMemoryDocumentRepository()
    source = record()

    repository.save(source)

    assert repository.get(source.document_id) == source
    assert repository.get("missing") is None
    assert repository.is_ready() is True

    assert repository.delete(source.document_id) is True
    assert repository.delete(source.document_id) is False
    assert repository.get(source.document_id) is None


def test_sql_repository_round_trips_and_replaces_chunks_atomically() -> None:
    repository = SqlDocumentRepository(create_engine("sqlite+pysqlite:///:memory:"))
    repository.create_schema()
    original = record()
    repository.save(original)

    replacement = DocumentRecord(
        **{
            **{field: getattr(original, field) for field in (
                "document_id", "filename", "content_type", "size_bytes",
                "sha256", "uploaded_at", "character_count"
            )},
            "chunks": (
                TextChunk(
                    index=0,
                    text="hello",
                    start_char=0,
                    end_char=5,
                    page_start=1,
                    page_end=1,
                ),
                TextChunk(
                    index=1,
                    text="world",
                    start_char=6,
                    end_char=11,
                    page_start=2,
                    page_end=2,
                ),
            ),
        }
    )
    repository.save(replacement)

    restored = repository.get("doc-1")
    assert restored is not None
    assert restored.chunks == replacement.chunks
    assert repository.is_ready() is True


def test_sql_repository_returns_none_for_unknown_document() -> None:
    repository = SqlDocumentRepository(create_engine("sqlite+pysqlite:///:memory:"))
    repository.create_schema()

    assert repository.get("missing") is None


def test_sql_repository_delete_removes_document_and_chunks_idempotently() -> None:
    repository = SqlDocumentRepository(create_engine("sqlite+pysqlite:///:memory:"))
    repository.create_schema()
    repository.save(record())

    assert repository.delete("doc-1") is True
    assert repository.delete("doc-1") is False
    assert repository.get("doc-1") is None
