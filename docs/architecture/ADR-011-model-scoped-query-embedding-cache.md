# ADR-011: Model-scoped query embedding cache

## Context / Problem

Search and question-answering often receive the same normalized query. Recomputing
its embedding adds avoidable remote-provider cost and latency, while caching without
the embedding contract can return a vector from an incompatible model or dimension.
Raw questions may also contain sensitive text and should not appear in cache keys or
metric labels.

## Decision

Put an `EmbeddingCache` boundary in the query path only. Use Redis with a finite TTL
when `REDIS_URL` is configured and a process-local in-memory implementation otherwise.
Keys contain the cache namespace, SHA-256 digests of the exact embedder model version
and stripped query, and vector dimensions. Validate every cached vector for exact
length, numeric type, and finite values before use; evict malformed Redis values.

Cache read or write errors do not fail a live search or question request: the service
records a bounded metric and computes the embedding normally. Readiness does fail
when a configured Redis backend cannot be reached, making degraded dependency state
visible to an orchestrator. Document-chunk embeddings are not cached because they are
already persisted by the vector-store boundary.

## Why this approach

The model-and-dimension key contract prevents cross-model vector reuse during a
deployment change. A narrow protocol keeps request orchestration independent of
Redis, preserves key-free local development, and makes failure behavior testable.
Hashed query keys reduce accidental disclosure in Redis inspection and telemetry.

## Alternatives considered

- **Cache search results:** rejected because results also depend on index contents,
  document scope, and limit, which require a broader invalidation contract.
- **Cache by query text alone:** rejected because model or dimension changes could
  silently return incompatible vectors.
- **Make Redis mandatory:** rejected because the local fallback should remain easy to
  run without infrastructure.
- **Fail requests when Redis fails:** rejected because embeddings remain computable;
  cache availability should not become data-plane availability.

## Tradeoffs / risks

In-memory entries are not shared across replicas and disappear on restart. Redis adds
an operational dependency, and hashed low-entropy questions remain vulnerable to
dictionary guessing, so Redis must stay on a private network. The TTL bounds stale
storage and memory use but can cause periodic recomputation. Readiness is deliberately
stricter than request handling: a Redis outage reports not-ready while in-flight
requests can still degrade safely to the provider.

## How it fits the architecture

The cache sits between HTTP query orchestration and the existing `Embedder` boundary.
It does not change ingestion, vector persistence, ranking, answer generation, or
citation provenance. `/ready` reports its backend and verifies configured Redis;
Prometheus counters expose hit, miss, success, and error outcomes without query text.

## Interview explanation

“Query embeddings are deterministic for a specific model contract, so I cache them
using keys scoped by model version and dimensions. Redis shares the cache across
replicas, while an in-memory adapter keeps local development simple. I validate cache
payloads as untrusted data, avoid raw questions in keys and labels, and let requests
fall back to live embedding if Redis fails while readiness still exposes the outage.”
