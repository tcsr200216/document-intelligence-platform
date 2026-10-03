from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

import httpx

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Boundary for converting text into fixed-size numeric vectors."""

    @property
    def dimensions(self) -> int:
        ...

    @property
    def provider(self) -> str:
        ...

    @property
    def model_version(self) -> str:
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class HashingEmbedder:
    """Deterministic local embedding fallback with no external model dependency.

    Tokens are hashed into a fixed-size feature vector and L2-normalized. This
    is intentionally a development baseline, not a semantic-embedding model.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero.")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provider(self) -> str:
        return "hashing"

    @property
    def model_version(self) -> str:
        return f"sha256-token-hashing-v1-{self.dimensions}d"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in _TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self._dimensions
            vector[bucket] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class EmbeddingProviderError(RuntimeError):
    """Raised when a configured remote embedding provider cannot return valid vectors."""


class OpenAIEmbedder:
    """OpenAI-compatible semantic embedding adapter with strict response validation."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1_536,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key is required for the OpenAI embedding provider.")
        if not model.strip():
            raise ValueError("model must not be empty.")
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._client = client or httpx.Client()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model_version(self) -> str:
        return f"{self._model}-{self.dimensions}d"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding input must contain readable text.")

        try:
            response = self._client.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "input": list(texts),
                    "model": self._model,
                    "dimensions": self.dimensions,
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingProviderError("Remote embedding request failed.") from exc

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingProviderError("Embedding response has an invalid item count.")

        by_index: dict[int, list[float]] = {}
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                raise EmbeddingProviderError("Embedding response is missing a valid index.")
            index = item["index"]
            vector = item.get("embedding")
            if index in by_index or not isinstance(vector, list):
                raise EmbeddingProviderError("Embedding response contains invalid vector data.")
            try:
                values = [float(value) for value in vector]
            except (TypeError, ValueError, OverflowError) as exc:
                raise EmbeddingProviderError("Embedding vector contains non-numeric values.") from exc
            if len(values) != self.dimensions or not all(math.isfinite(value) for value in values):
                raise EmbeddingProviderError(
                    "Embedding vector dimensions or numeric values are invalid."
                )
            by_index[index] = values

        expected_indices = set(range(len(texts)))
        if set(by_index) != expected_indices:
            raise EmbeddingProviderError("Embedding response indices are incomplete.")
        return [by_index[index] for index in range(len(texts))]


def build_embedder(
    *,
    provider: str,
    dimensions: int,
    openai_api_key: str | None = None,
    openai_model: str = "text-embedding-3-small",
    openai_base_url: str = "https://api.openai.com/v1",
    timeout_seconds: float = 20.0,
) -> Embedder:
    """Build the configured provider while keeping local development key-free."""
    if provider == "hashing":
        return HashingEmbedder(dimensions=dimensions)
    if provider == "openai":
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai.")
        return OpenAIEmbedder(
            api_key=openai_api_key,
            model=openai_model,
            dimensions=dimensions,
            base_url=openai_base_url,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")
