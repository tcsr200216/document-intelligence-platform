import math

import httpx
import pytest

from app.embeddings import (
    EmbeddingProviderError,
    HashingEmbedder,
    OpenAIEmbedder,
    build_embedder,
)


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


def test_openai_embedder_sends_batch_and_restores_provider_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://embeddings.example/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.read().decode() == (
            '{"input":["first","second"],"model":"semantic-test","dimensions":3}'
        )
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0, 0.0]},
                    {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                ]
            },
        )

    embedder = OpenAIEmbedder(
        api_key="test-key",
        model="semantic-test",
        dimensions=3,
        base_url="https://embeddings.example/v1/",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert embedder.provider == "openai"
    assert embedder.model_version == "semantic-test-3d"
    assert embedder.embed(["first", "second"]) == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]


def test_openai_embedder_rejects_wrong_dimensions() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]}
        )
    )
    embedder = OpenAIEmbedder(
        api_key="test-key",
        dimensions=3,
        client=httpx.Client(transport=transport),
    )

    with pytest.raises(EmbeddingProviderError, match="dimensions"):
        embedder.embed(["valid text"])


def test_openai_embedder_wraps_remote_failure_without_exposing_response() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(429, text="sensitive body"))
    embedder = OpenAIEmbedder(
        api_key="test-key",
        dimensions=3,
        client=httpx.Client(transport=transport),
    )

    with pytest.raises(EmbeddingProviderError, match="Remote embedding request failed") as error:
        embedder.embed(["valid text"])

    assert "sensitive body" not in str(error.value)


def test_builder_keeps_hashing_default_and_requires_key_for_openai() -> None:
    local = build_embedder(provider="hashing", dimensions=32)

    assert isinstance(local, HashingEmbedder)
    assert local.model_version == "sha256-token-hashing-v1-32d"
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_embedder(provider="openai", dimensions=32)
