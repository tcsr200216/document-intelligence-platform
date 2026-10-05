import json

from app.cache import InMemoryEmbeddingCache, RedisEmbeddingCache


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.deleted: list[str] = []

    def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        assert ttl > 0
        self.values[key] = value.encode()

    def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.values.pop(key, None)

    def ping(self) -> bool:
        return True


def test_memory_cache_expires_and_scopes_entries_by_model_and_dimensions() -> None:
    now = [100.0]
    cache = InMemoryEmbeddingCache(ttl_seconds=10, clock=lambda: now[0])
    cache.set("same query", [0.25, 0.75], model_version="model-v1", dimensions=2)

    assert cache.get("same query", model_version="model-v1", dimensions=2) == (
        0.25,
        0.75,
    )
    assert cache.get("same query", model_version="model-v2", dimensions=2) is None
    assert cache.get("same query", model_version="model-v1", dimensions=3) is None

    now[0] = 110.0
    assert cache.get("same query", model_version="model-v1", dimensions=2) is None


def test_memory_cache_rejects_invalid_vectors() -> None:
    cache = InMemoryEmbeddingCache()

    try:
        cache.set("query", [1.0], model_version="model-v1", dimensions=2)
    except ValueError as exc:
        assert "dimensions" in str(exc)
    else:
        raise AssertionError("dimension mismatch should be rejected")


def test_redis_cache_uses_hashed_keys_and_round_trips_valid_vectors() -> None:
    cache = RedisEmbeddingCache("redis://unused", ttl_seconds=45)
    fake = FakeRedis()
    cache._client = fake  # type: ignore[assignment]

    cache.set("private customer question", [0.1, 0.9], model_version="model-v1", dimensions=2)

    key = next(iter(fake.values))
    assert "private customer question" not in key
    assert key.startswith("query-embeddings:v1:")
    assert "model-v1" not in key
    assert ":2:" in key
    assert cache.get("private customer question", model_version="model-v1", dimensions=2) == (
        0.1,
        0.9,
    )


def test_redis_cache_evicts_malformed_or_wrong_dimension_values() -> None:
    cache = RedisEmbeddingCache("redis://unused")
    fake = FakeRedis()
    cache._client = fake  # type: ignore[assignment]
    cache.set("query", [0.1, 0.9], model_version="model-v1", dimensions=2)
    key = next(iter(fake.values))
    fake.values[key] = json.dumps([0.5]).encode()

    assert cache.get("query", model_version="model-v1", dimensions=2) is None
    assert fake.deleted == [key]
