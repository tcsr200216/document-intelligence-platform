from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from redis import Redis
from redis.exceptions import RedisError

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class EmbeddingCache(Protocol):
    """Cache boundary for model-scoped query embeddings."""

    @property
    def backend(self) -> str: ...

    def get(
        self, text: str, *, model_version: str, dimensions: int
    ) -> tuple[float, ...] | None: ...

    def set(
        self,
        text: str,
        vector: Sequence[float],
        *,
        model_version: str,
        dimensions: int,
    ) -> None: ...

    def is_ready(self) -> bool: ...


def _validate_identifier(value: str, name: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be 1-128 URL-safe characters.")
    return value


def _cache_key(namespace: str, text: str, model_version: str, dimensions: int) -> str:
    namespace = _validate_identifier(namespace, "Cache namespace")
    if not model_version.strip() or len(model_version) > 512:
        raise ValueError("Embedding model version must be 1-512 characters.")
    if dimensions < 1:
        raise ValueError("Embedding dimensions must be positive.")
    model_digest = hashlib.sha256(model_version.encode("utf-8")).hexdigest()
    query_digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    return f"{namespace}:{model_digest}:{dimensions}:{query_digest}"


def _validated_vector(value: object, dimensions: int) -> tuple[float, ...] | None:
    if not isinstance(value, list) or len(value) != dimensions:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    vector = tuple(float(item) for item in value)
    return vector if all(math.isfinite(item) for item in vector) else None


class InMemoryEmbeddingCache:
    def __init__(
        self,
        *,
        ttl_seconds: int = 300,
        namespace: str = "query-embeddings:v1",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError("Cache TTL must be positive.")
        self._ttl_seconds = ttl_seconds
        self._namespace = _validate_identifier(namespace, "Cache namespace")
        self._clock = clock
        self._entries: dict[str, tuple[float, tuple[float, ...]]] = {}

    @property
    def backend(self) -> str:
        return "memory"

    def get(self, text: str, *, model_version: str, dimensions: int) -> tuple[float, ...] | None:
        key = _cache_key(self._namespace, text, model_version, dimensions)
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, vector = entry
        if expires_at <= self._clock():
            self._entries.pop(key, None)
            return None
        return vector

    def set(
        self,
        text: str,
        vector: Sequence[float],
        *,
        model_version: str,
        dimensions: int,
    ) -> None:
        validated = _validated_vector(list(vector), dimensions)
        if validated is None:
            raise ValueError("Cached embedding does not match the configured dimensions.")
        key = _cache_key(self._namespace, text, model_version, dimensions)
        self._entries[key] = (self._clock() + self._ttl_seconds, validated)

    def is_ready(self) -> bool:
        return True


class RedisEmbeddingCache:
    def __init__(
        self,
        redis_url: str,
        *,
        ttl_seconds: int = 300,
        namespace: str = "query-embeddings:v1",
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError("Cache TTL must be positive.")
        self._client = Redis.from_url(redis_url)
        self._ttl_seconds = ttl_seconds
        self._namespace = _validate_identifier(namespace, "Cache namespace")

    @property
    def backend(self) -> str:
        return "redis"

    def get(self, text: str, *, model_version: str, dimensions: int) -> tuple[float, ...] | None:
        key = _cache_key(self._namespace, text, model_version, dimensions)
        payload = self._client.get(key)
        if payload is None:
            return None
        try:
            decoded = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            self._client.delete(key)
            return None
        vector = _validated_vector(decoded, dimensions)
        if vector is None:
            self._client.delete(key)
        return vector

    def set(
        self,
        text: str,
        vector: Sequence[float],
        *,
        model_version: str,
        dimensions: int,
    ) -> None:
        validated = _validated_vector(list(vector), dimensions)
        if validated is None:
            raise ValueError("Cached embedding does not match the configured dimensions.")
        key = _cache_key(self._namespace, text, model_version, dimensions)
        self._client.setex(key, self._ttl_seconds, json.dumps(validated, separators=(",", ":")))

    def is_ready(self) -> bool:
        try:
            return bool(self._client.ping())
        except RedisError:
            return False


def build_embedding_cache(
    redis_url: str | None,
    *,
    ttl_seconds: int,
    namespace: str,
) -> EmbeddingCache:
    if redis_url:
        return RedisEmbeddingCache(
            redis_url,
            ttl_seconds=ttl_seconds,
            namespace=namespace,
        )
    return InMemoryEmbeddingCache(ttl_seconds=ttl_seconds, namespace=namespace)
