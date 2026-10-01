from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.parsing import TextChunk


@dataclass(frozen=True, slots=True)
class StoredChunk:
    """A source chunk and its corresponding embedding."""

    document_id: str
    chunk: TextChunk
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A retrieved source span with its cosine-similarity score."""

    document_id: str
    chunk: TextChunk
    score: float


class VectorStore(Protocol):
    """Index/retrieval boundary independent of the embedding provider."""

    @property
    def dimensions(self) -> int:
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
