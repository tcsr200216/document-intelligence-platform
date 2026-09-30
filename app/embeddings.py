from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Boundary for converting text into fixed-size numeric vectors."""

    @property
    def dimensions(self) -> int:
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
