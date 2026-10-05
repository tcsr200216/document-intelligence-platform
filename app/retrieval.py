from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.vector_store import SearchHit, VectorStore


@dataclass(frozen=True, slots=True)
class HybridRetriever:
    """Fuse semantic and lexical rankings without coupling either storage backend."""

    vector_store: VectorStore
    semantic_weight: float = 0.65
    lexical_weight: float = 0.35
    reciprocal_rank_constant: int = 60
    candidate_multiplier: int = 4

    def __post_init__(self) -> None:
        if self.semantic_weight <= 0 or self.lexical_weight <= 0:
            raise ValueError("Retrieval weights must be greater than zero.")
        if self.reciprocal_rank_constant < 0:
            raise ValueError("reciprocal_rank_constant must not be negative.")
        if self.candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than zero.")

    def retrieve(
        self,
        query: str,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")
        candidate_limit = limit * self.candidate_multiplier
        semantic = self.vector_store.search(
            query_vector,
            limit=candidate_limit,
            document_id=document_id,
        )
        lexical = self.vector_store.lexical_search(
            query,
            limit=candidate_limit,
            document_id=document_id,
        )

        by_key: dict[tuple[str, int], SearchHit] = {}
        scores: dict[tuple[str, int], float] = {}
        for hits, weight in (
            (semantic, self.semantic_weight),
            (lexical, self.lexical_weight),
        ):
            for rank, hit in enumerate(hits, start=1):
                key = (hit.document_id, hit.chunk.index)
                by_key.setdefault(key, hit)
                scores[key] = scores.get(key, 0.0) + weight / (
                    self.reciprocal_rank_constant + rank
                )

        maximum = (self.semantic_weight + self.lexical_weight) / (
            self.reciprocal_rank_constant + 1
        )
        ordered = sorted(
            scores,
            key=lambda key: (-scores[key], key[0], key[1]),
        )[:limit]
        return [
            SearchHit(
                document_id=by_key[key].document_id,
                chunk=by_key[key].chunk,
                score=min(1.0, scores[key] / maximum),
            )
            for key in ordered
        ]
