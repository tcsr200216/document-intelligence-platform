import math

import pytest

from app.parsing import TextChunk
from app.vector_store import InMemoryVectorStore, build_vector_store


def span(index: int, text: str, start: int = 0) -> TextChunk:
    return TextChunk(index=index, text=text, start_char=start, end_char=start + len(text))


def test_search_ranks_by_cosine_and_preserves_source_provenance() -> None:
    store = InMemoryVectorStore(dimensions=2)
    source = span(3, "FastAPI handles requests", start=120)
    other = span(0, "Unrelated section", start=0)
    store.replace_document("doc-b", [source, other], [[2.0, 0.0], [0.0, 3.0]])

    hits = store.search([1.0, 0.0])

    assert [hit.chunk for hit in hits] == [source, other]
    assert hits[0].document_id == "doc-b"
    assert hits[0].chunk.start_char == 120
    assert hits[0].chunk.end_char == 120 + len(source.text)
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.0)


def test_search_can_be_scoped_to_one_document() -> None:
    store = InMemoryVectorStore(dimensions=2)
    store.replace_document("doc-a", [span(0, "alpha")], [[1, 0]])
    store.replace_document("doc-b", [span(0, "beta")], [[1, 0]])

    assert [hit.document_id for hit in store.search([1, 0], document_id="doc-b")] == [
        "doc-b"
    ]
    assert store.search([1, 0], document_id="missing") == []


def test_equal_scores_have_stable_document_and_chunk_order() -> None:
    store = InMemoryVectorStore(dimensions=2)
    store.replace_document("doc-z", [span(2, "z")], [[1, 0]])
    store.replace_document(
        "doc-a", [span(3, "a3"), span(1, "a1")], [[1, 0], [1, 0]]
    )

    assert [(hit.document_id, hit.chunk.index) for hit in store.search([1, 0])] == [
        ("doc-a", 1), ("doc-a", 3), ("doc-z", 2)
    ]
    assert len(store.search([1, 0], limit=1)) == 1


def test_replacing_document_removes_old_chunks() -> None:
    store = InMemoryVectorStore(dimensions=2)
    store.replace_document("doc", [span(0, "old")], [[1, 0]])
    store.replace_document("doc", [span(1, "new")], [[0, 1]])

    assert [hit.chunk.text for hit in store.search([0, 1])] == ["new"]
    store.replace_document("doc", [], [])
    assert store.search([0, 1]) == []


def test_invalid_replacement_does_not_mutate_existing_document() -> None:
    store = InMemoryVectorStore(dimensions=2)
    store.replace_document("doc", [span(0, "original")], [[1, 0]])

    with pytest.raises(ValueError, match="dimension mismatch"):
        store.replace_document("doc", [span(0, "replacement")], [[1]])

    assert [hit.chunk.text for hit in store.search([1, 0])] == ["original"]


@pytest.mark.parametrize("dimension", [0, -1])
def test_rejects_invalid_store_dimensions(dimension: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        InMemoryVectorStore(dimensions=dimension)


def test_rejects_mismatched_vector_counts_and_duplicate_chunk_indices() -> None:
    store = InMemoryVectorStore(dimensions=2)
    with pytest.raises(ValueError, match="exactly one embedding"):
        store.replace_document("doc", [span(0, "one")], [])
    with pytest.raises(ValueError, match="unique"):
        store.replace_document("doc", [span(0, "one"), span(0, "two")], [[1, 0], [0, 1]])


@pytest.mark.parametrize("vector", [[1.0], [1.0, math.nan], [1.0, math.inf]])
def test_rejects_invalid_embedding_vectors(vector: list[float]) -> None:
    store = InMemoryVectorStore(dimensions=2)
    with pytest.raises(ValueError, match="dimension mismatch|finite numbers"):
        store.replace_document("doc", [span(0, "text")], [vector])
    with pytest.raises(ValueError, match="dimension mismatch|finite numbers"):
        store.search(vector)


def test_zero_query_or_zero_index_vector_has_no_relevance_signal() -> None:
    store = InMemoryVectorStore(dimensions=2)
    store.replace_document("doc", [span(0, "empty")], [[0, 0]])
    assert store.search([1, 0]) == []
    assert store.search([0, 0]) == []


def test_invalid_query_limit_and_document_id_are_rejected() -> None:
    store = InMemoryVectorStore(dimensions=2)
    with pytest.raises(ValueError, match="limit"):
        store.search([1, 0], limit=0)
    with pytest.raises(ValueError, match="document_id filter"):
        store.search([1, 0], document_id=" ")
    with pytest.raises(ValueError, match="document_id"):
        store.replace_document(" ", [], [])


def test_builder_keeps_key_free_local_store_without_database() -> None:
    store = build_vector_store(None, dimensions=2, model_version="test-v1-2d")
    assert isinstance(store, InMemoryVectorStore)
    assert store.backend == "memory"
    assert store.is_ready()
