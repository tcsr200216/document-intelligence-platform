from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    Column,
    Computed,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.parsing import TextChunk


@dataclass(frozen=True, slots=True)
class StoredChunk:
    """A source chunk and its corresponding embedding."""

    document_id: str
    chunk: TextChunk
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A retrieved source span and its backend-specific relevance score."""

    document_id: str
    chunk: TextChunk
    score: float


class VectorStore(Protocol):
    """Index/retrieval boundary independent of the embedding provider."""

    @property
    def dimensions(self) -> int:
        ...

    @property
    def backend(self) -> str:
        ...

    def is_ready(self) -> bool:
        ...

    def replace_document(
        self,
        document_id: str,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Atomically replace all indexed chunks for one document."""
        ...

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        """Find the best matching source spans."""
        ...

    def lexical_search(
        self,
        query: str,
        *,
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        """Find source spans containing the query's lexical evidence."""
        ...

    def delete_document(self, document_id: str) -> None:
        """Idempotently remove every indexed chunk for one document."""
        ...


_LEXICAL_TOKEN = re.compile(r"[\w]+", re.UNICODE)


def _lexical_tokens(text_value: str) -> tuple[str, ...]:
    return tuple(_LEXICAL_TOKEN.findall(text_value.casefold()))


class InMemoryVectorStore:
    """Deterministic, process-local cosine retriever for development and tests.

    It does not persist across restarts or offer approximate nearest-neighbor indexing.
    """

    def __init__(self, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero.")
        self._dimensions = dimensions
        self._documents: dict[str, tuple[StoredChunk, ...]] = {}

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def backend(self) -> str:
        return "memory"

    def is_ready(self) -> bool:
        return True

    def _validate_vector(self, vector: Sequence[float]) -> tuple[float, ...]:
        if len(vector) != self._dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimensions}, got {len(vector)}."
            )
        try:
            values = tuple(float(value) for value in vector)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Embedding values must be finite numbers.") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Embedding values must be finite numbers.")
        return values

    def replace_document(
        self,
        document_id: str,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if not document_id or not document_id.strip():
            raise ValueError("document_id must not be empty.")
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding.")
        if len({chunk.index for chunk in chunks}) != len(chunks):
            raise ValueError("Chunk indices must be unique within a document.")

        # Validate everything before changing the index: a failed replacement
        # must never leave a previously indexed document half-updated.
        records = tuple(
            StoredChunk(document_id, chunk, self._validate_vector(vector))
            for chunk, vector in zip(chunks, vectors, strict=True)
        )
        self._documents[document_id] = records

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")
        if document_id is not None and not document_id.strip():
            raise ValueError("document_id filter must not be empty.")

        query = self._validate_vector(query_vector)
        query_norm = math.sqrt(math.fsum(value * value for value in query))
        if query_norm == 0:
            return []

        documents = (
            ((document_id, self._documents.get(document_id, ())),)
            if document_id is not None
            else self._documents.items()
        )
        hits: list[SearchHit] = []
        for stored_document_id, records in documents:
            for record in records:
                record_norm = math.sqrt(math.fsum(value * value for value in record.vector))
                if record_norm == 0:
                    continue
                score = math.fsum(
                    left * right for left, right in zip(query, record.vector, strict=True)
                ) / (query_norm * record_norm)
                hits.append(
                    SearchHit(
                        document_id=stored_document_id,
                        chunk=record.chunk,
                        score=max(-1.0, min(1.0, score)),
                    )
                )

        return sorted(
            hits,
            key=lambda hit: (-hit.score, hit.document_id, hit.chunk.index),
        )[:limit]

    def lexical_search(
        self,
        query: str,
        *,
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")
        if document_id is not None and not document_id.strip():
            raise ValueError("document_id filter must not be empty.")
        query_tokens = _lexical_tokens(query)
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        documents = (
            ((document_id, self._documents.get(document_id, ())),)
            if document_id is not None
            else self._documents.items()
        )
        hits: list[SearchHit] = []
        for stored_document_id, records in documents:
            for record in records:
                chunk_counts = Counter(_lexical_tokens(record.chunk.text))
                matched = sum(
                    min(count, chunk_counts[token]) for token, count in query_counts.items()
                )
                if matched == 0:
                    continue
                hits.append(
                    SearchHit(
                        document_id=stored_document_id,
                        chunk=record.chunk,
                        score=matched / len(query_tokens),
                    )
                )
        return sorted(
            hits,
            key=lambda hit: (-hit.score, hit.document_id, hit.chunk.index),
        )[:limit]

    def delete_document(self, document_id: str) -> None:
        if not document_id or not document_id.strip():
            raise ValueError("document_id must not be empty.")
        self._documents.pop(document_id, None)


class PgVectorStore:
    """Durable PostgreSQL cosine retrieval scoped to one embedding model."""

    def __init__(self, engine: Engine, dimensions: int, model_version: str) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero.")
        if not model_version.strip():
            raise ValueError("model_version must not be empty.")
        self._engine = engine
        self._dimensions = dimensions
        self._model_version = model_version
        self._metadata = MetaData()
        self._config = Table(
            "vector_store_config",
            self._metadata,
            Column("id", Integer, primary_key=True),
            Column("dimensions", Integer, nullable=False),
            Column("model_version", String(256), nullable=False),
        )
        self._vectors = Table(
            "document_vectors",
            self._metadata,
            Column("document_id", String(36), primary_key=True),
            Column("chunk_index", Integer, primary_key=True),
            Column("model_version", String(256), primary_key=True),
            Column("text", Text, nullable=False),
            Column("start_char", Integer, nullable=False),
            Column("end_char", Integer, nullable=False),
            Column("page_start", Integer, nullable=True),
            Column("page_end", Integer, nullable=True),
            Column("embedding", VECTOR(dimensions), nullable=False),
            Column(
                "search_vector",
                TSVECTOR,
                Computed("to_tsvector('english', text)", persisted=True),
                nullable=False,
            ),
        )
        Index(
            "ix_document_vectors_search_vector_gin",
            self._vectors.c.search_vector,
            postgresql_using="gin",
        )
        # pgvector's vector HNSW index supports up to 2,000 dimensions. Higher
        # dimensional providers remain durable and use exact database-side scans.
        if dimensions <= 2_000:
            Index(
                "ix_document_vectors_embedding_hnsw",
                self._vectors.c.embedding,
                postgresql_using="hnsw",
                postgresql_ops={"embedding": "vector_cosine_ops"},
            )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def backend(self) -> str:
        return "pgvector"

    @classmethod
    def from_url(
        cls, database_url: str, dimensions: int, model_version: str
    ) -> PgVectorStore:
        store = cls(
            create_engine(database_url, pool_pre_ping=True),
            dimensions,
            model_version,
        )
        store.create_schema()
        return store

    def create_schema(self) -> None:
        with self._engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        self._metadata.create_all(self._engine)
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE document_vectors "
                    "ADD COLUMN IF NOT EXISTS page_start INTEGER"
                )
            )
            connection.execute(
                text(
                    "ALTER TABLE document_vectors "
                    "ADD COLUMN IF NOT EXISTS page_end INTEGER"
                )
            )
            connection.execute(
                text(
                    "ALTER TABLE document_vectors ADD COLUMN IF NOT EXISTS search_vector "
                    "tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_document_vectors_search_vector_gin "
                    "ON document_vectors USING gin (search_vector)"
                )
            )
            configured = connection.execute(select(self._config)).mappings().one_or_none()
            expected = {
                "id": 1,
                "dimensions": self.dimensions,
                "model_version": self._model_version,
            }
            if configured is None:
                connection.execute(insert(self._config).values(**expected))
            elif dict(configured) != expected:
                raise ValueError(
                    "The durable vector index uses a different embedding model or dimension. "
                    "Rebuild the index before changing embedding configuration."
                )

    def _validate_vector(self, vector: Sequence[float]) -> list[float]:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimensions}, got {len(vector)}."
            )
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Embedding values must be finite numbers.") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Embedding values must be finite numbers.")
        return values

    def replace_document(
        self,
        document_id: str,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if not document_id or not document_id.strip():
            raise ValueError("document_id must not be empty.")
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding.")
        if len({chunk.index for chunk in chunks}) != len(chunks):
            raise ValueError("Chunk indices must be unique within a document.")
        rows = [
            {
                "document_id": document_id,
                "chunk_index": chunk.index,
                "model_version": self._model_version,
                "text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "embedding": self._validate_vector(vector),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        with self._engine.begin() as connection:
            connection.execute(
                delete(self._vectors).where(
                    self._vectors.c.document_id == document_id,
                    self._vectors.c.model_version == self._model_version,
                )
            )
            if rows:
                connection.execute(insert(self._vectors), rows)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")
        if document_id is not None and not document_id.strip():
            raise ValueError("document_id filter must not be empty.")
        query = self._validate_vector(query_vector)
        if math.fsum(value * value for value in query) == 0:
            return []
        distance = self._vectors.c.embedding.cosine_distance(query)
        statement = (
            select(self._vectors, distance.label("distance"))
            .where(self._vectors.c.model_version == self._model_version)
            .where(distance.is_not(None))
        )
        if document_id is not None:
            statement = statement.where(self._vectors.c.document_id == document_id)
        statement = statement.order_by(
            distance,
            self._vectors.c.document_id,
            self._vectors.c.chunk_index,
        ).limit(limit)
        with self._engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [
            SearchHit(
                document_id=row["document_id"],
                chunk=TextChunk(
                    index=row["chunk_index"],
                    text=row["text"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                ),
                score=max(-1.0, min(1.0, 1.0 - float(row["distance"]))),
            )
            for row in rows
        ]

    def lexical_search(
        self,
        query: str,
        *,
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")
        if document_id is not None and not document_id.strip():
            raise ValueError("document_id filter must not be empty.")
        if not _lexical_tokens(query):
            return []

        text_query = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank_cd(self._vectors.c.search_vector, text_query)
        statement = (
            select(self._vectors, rank.label("rank"))
            .where(self._vectors.c.model_version == self._model_version)
            .where(self._vectors.c.search_vector.op("@@")(text_query))
        )
        if document_id is not None:
            statement = statement.where(self._vectors.c.document_id == document_id)
        statement = statement.order_by(
            rank.desc(),
            self._vectors.c.document_id,
            self._vectors.c.chunk_index,
        ).limit(limit)
        with self._engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [
            SearchHit(
                document_id=row["document_id"],
                chunk=TextChunk(
                    index=row["chunk_index"],
                    text=row["text"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                ),
                score=float(row["rank"]),
            )
            for row in rows
        ]

    def delete_document(self, document_id: str) -> None:
        if not document_id or not document_id.strip():
            raise ValueError("document_id must not be empty.")
        with self._engine.begin() as connection:
            connection.execute(
                delete(self._vectors).where(
                    self._vectors.c.document_id == document_id,
                    self._vectors.c.model_version == self._model_version,
                )
            )

    def is_ready(self) -> bool:
        try:
            with self._engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False


def build_vector_store(
    database_url: str | None,
    *,
    dimensions: int,
    model_version: str,
) -> VectorStore:
    if not database_url:
        return InMemoryVectorStore(dimensions)
    return PgVectorStore.from_url(database_url, dimensions, model_version)
