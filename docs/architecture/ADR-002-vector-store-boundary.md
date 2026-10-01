# ADR-002: Vector-store boundary and deterministic local cosine retrieval

## Context / Problem

The parsing and chunking stages provide text with exact character offsets, and the
embedding provider produces fixed-dimensional vectors. Retrieval needs to rank those
vectors without coupling ingestion, citation provenance, or API design to a specific
database. A local backend must also run without cloud credentials or infrastructure.

## Decision

Define a `VectorStore` protocol with document replacement and optional document-scoped
search. Implement an `InMemoryVectorStore` that indexes immutable source chunks alongside
their vectors and scores candidates with cosine similarity. Enforce a store-wide embedding
dimension, validate finite numeric values, ignore zero-length vectors during ranking, and
break equal-score ties by document ID and chunk index. Replacing a document validates its
new records fully before swapping them into the index.

## Why this approach

A deterministic exact-search implementation is small and useful for integration tests and
local demonstrations. Storing each original `TextChunk`, including its `start_char` and
`end_char`, retains the trace needed for citation-backed responses. The provider and
storage boundaries can evolve independently: a real semantic embedder can provide the
vectors and a durable vector backend can implement the same store contract.

## Alternatives considered

- Store vectors directly in API handlers: fewer files, but couples the HTTP layer to
  retrieval and complicates testing.
- Require pgvector from day one: supports durable indexed retrieval, but makes the local
  development path depend on a configured PostgreSQL instance.
- Use a dedicated vector service immediately: provides scalability options at the cost
  of additional deployment complexity.
- Use raw dot product: works for normalized vectors but gives misleading rankings when
  providers produce vectors with different magnitudes.

## Tradeoffs / risks

The in-memory backend is process-local, loses data on restart, and performs an O(N × D)
exact scan, so it is a correctness/development baseline rather than a scalable production
index. It does not perform metadata persistence, authentication, or query generation.
HashingEmbedder remains a token-overlap baseline, not a semantic model. A future pgvector
adapter will need embedding-model/version consistency, schema/migration management, and
a strategy for atomically persisting document metadata and vectors.

## How it fits the architecture

Normalized source text → traceable `TextChunk` spans → `Embedder` vectors →
`VectorStore.replace_document` → `VectorStore.search` → source-aware search hits.
Future answer generation can cite the returned document ID, text, and source offsets
without depending on which vector backend produced the hits.

## Interview explanation

"I introduced a vector-store port instead of putting similarity logic in the API. Its
local adapter uses deterministic cosine retrieval and returns original chunk offsets,
which we need for verifiable citations. I validate dimensions and replace a document
atomically within the process, so changing embedding providers cannot silently corrupt
the index. This is intentionally an exact in-memory baseline; for hosted scale I'd add
a durable pgvector adapter and migrations behind the same interface."
