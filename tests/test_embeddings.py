import math

import pytest

from app.embeddings import HashingEmbedder


def test_hashing_embedder_is_deterministic() -> None:
    embedder = HashingEmbedder(dimensions=32)

    first = embedder.embed(["Chunking preserves citation offsets."])[0]
    second = embedder.embed(["Chunking preserves citation offsets."])[0]

    assert first == second
    assert len(first) == 32


def test_hashing_embedder_is_case_insensitive_and_normalized() -> None:
    embedder = HashingEmbedder(dimensions=64)

    lower, upper = embedder.embed(["retrieval pipeline", "RETRIEVAL PIPELINE"])

    assert lower == upper
    assert math.sqrt(sum(value * value for value in lower)) == pytest.approx(1.0)


def test_hashing_embedder_returns_zero_vector_for_text_without_tokens() -> None:
    vector = HashingEmbedder(dimensions=8).embed(["--- !!!"])[0]

    assert vector == [0.0] * 8


def test_hashing_embedder_preserves_batch_order() -> None:
    embedder = HashingEmbedder(dimensions=16)

    batch = embedder.embed(["alpha", "beta", "alpha"])

    assert batch[0] == batch[2]
    assert batch[0] != batch[1]


def test_hashing_embedder_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        HashingEmbedder(dimensions=0)
