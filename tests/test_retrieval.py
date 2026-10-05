import pytest

from app.parsing import TextChunk
from app.retrieval import HybridRetriever
from app.vector_store import InMemoryVectorStore


def chunk(index: int, text: str) -> TextChunk:
    return TextChunk(index=index, text=text, start_char=index * 100, end_char=index * 100 + len(text))


def test_hybrid_retrieval_rescues_exact_evidence_from_semantic_second_place() -> None:
    store = InMemoryVectorStore(dimensions=2)
    semantic_only = chunk(0, "general storage overview")
    exact_evidence = chunk(1, "invoice retention is seven years")
    store.replace_document(
        "policy",
        [semantic_only, exact_evidence],
        [[1.0, 0.0], [0.8, 0.2]],
    )

    hits = HybridRetriever(store).retrieve(
        "invoice retention seven years",
        [1.0, 0.0],
        limit=2,
    )

    assert [hit.chunk for hit in hits] == [exact_evidence, semantic_only]
    assert hits[0].score == pytest.approx(0.9898, rel=1e-3)
    assert hits[0].chunk.start_char == 100


def test_hybrid_retrieval_is_scoped_deduplicated_and_deterministic() -> None:
    store = InMemoryVectorStore(dimensions=2)
    store.replace_document("doc-a", [chunk(0, "alpha exact")], [[1, 0]])
    store.replace_document("doc-b", [chunk(0, "alpha exact")], [[1, 0]])

    hits = HybridRetriever(store).retrieve(
        "alpha exact",
        [1, 0],
        limit=3,
        document_id="doc-b",
    )

    assert [(hit.document_id, hit.chunk.index) for hit in hits] == [("doc-b", 0)]
    assert hits[0].score == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"semantic_weight": 0}, "weights"),
        ({"lexical_weight": 0}, "weights"),
        ({"reciprocal_rank_constant": -1}, "reciprocal_rank_constant"),
        ({"candidate_multiplier": 0}, "candidate_multiplier"),
    ],
)
def test_hybrid_retrieval_rejects_invalid_configuration(
    kwargs: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        HybridRetriever(InMemoryVectorStore(2), **kwargs)


def test_hybrid_retrieval_rejects_non_positive_limit() -> None:
    retriever = HybridRetriever(InMemoryVectorStore(2))
    with pytest.raises(ValueError, match="limit"):
        retriever.retrieve("query", [1, 0], limit=0)
