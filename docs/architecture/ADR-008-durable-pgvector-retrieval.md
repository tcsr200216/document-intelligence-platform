# ADR-008: Durable model-scoped retrieval with pgvector

## Context / Problem

PostgreSQL already preserved document metadata and exact citation spans, but embeddings
lived only in API memory. A restart made every persisted document unsearchable and
uncitable until it was uploaded again, and multiple replicas would have independent
indexes.

## Decision

When `DATABASE_URL` is configured, use a PostgreSQL pgvector adapter behind the existing
`VectorStore` boundary. Persist each vector with its document ID, chunk index, exact text,
character offsets, and embedding model version. Rank with database-side cosine distance
and an HNSW cosine index when dimensions are within pgvector's 2,000-dimension vector
index limit; use an exact database scan above it. Record the model version and dimensions in a singleton
configuration row and refuse startup on mismatch. Keep the deterministic in-memory store
when no database is configured.

CI uploads a document, verifies retrieval and cited Q&A, restarts the API while retaining
the database volume, and verifies the same document again.

## Why this approach

pgvector keeps metadata and retrieval in the existing operational database while providing
native vector types, cosine operators, and an approximate index. The adapter preserves the
port already used by the API and returns the same citation-bearing `SearchHit` objects.
Failing closed on model mismatch prevents semantically incompatible vectors from being
queried together.

## Alternatives considered

- **Rebuild the in-memory index at startup:** avoids an extension but requires re-embedding
  every document, delays readiness, and can call a paid remote provider after each restart.
- **Store vectors as JSON and score in Python:** durable but transfers the whole corpus and
  remains an O(N × D) application-side scan.
- **Use a dedicated vector database:** viable at larger scale, but adds another service and
  consistency boundary before this project's needs justify it.
- **Silently create a new namespace for every model:** keeps old data, but hides operational
  mistakes and leaves documents partly searchable. An explicit rebuild is safer.

## Tradeoffs / risks

Hosted PostgreSQL must allow the pgvector extension. HNSW consumes additional memory and
build time and trades exact exhaustive ranking for approximate retrieval at larger scale.
Vectors above 2,000 dimensions remain durable but fall back to a slower exact scan.
Document metadata and vectors are written in consecutive transactions through separate
ports, so a vector-write failure can leave metadata without an index entry; the upload fails
and can be retried, but a future ingestion job/outbox should reconcile partial writes.
Changing model or dimensions requires an intentional vector-index rebuild.

## How it fits the architecture

Parsing and chunking still own source offsets, the embedder still owns vector creation, and
the API still depends only on `VectorStore`. PostgreSQL-backed deployments now share one
durable retrieval index across restarts and replicas, while local key-free development keeps
the same in-memory implementation.

## Interview explanation

"The metadata was durable but the retrieval path was not, so a restart broke the product's
core promise. I added pgvector behind the existing port, scoped the index to an explicit
embedding contract, and proved durability by restarting the API in CI before repeating
search and citation-backed Q&A. I also documented the remaining cross-store transaction
gap instead of claiming exactly-once ingestion."
