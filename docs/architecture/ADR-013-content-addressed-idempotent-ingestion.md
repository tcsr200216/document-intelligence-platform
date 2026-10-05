# ADR-013: Content-addressed idempotent document ingestion

## Context / Problem

Upload clients retry when connections time out or responses are lost. Previously, the
same bytes produced a new document ID, repeated parsing and embedding costs, and added
duplicate chunks to retrieval. Duplicate citations make answers noisy, while a simple
application-level lookup alone cannot prevent two API replicas from ingesting the same
content concurrently.

## Decision

Treat the SHA-256 digest of the original uploaded bytes as the document's content
identity. The repository exposes digest lookup, enforces a unique digest in PostgreSQL,
and stores a publication status. A new document is first persisted as `indexing`; only
after all vectors are durable is it changed to `indexed`.

The upload path checks the digest before parsing. An existing indexed digest returns
HTTP `200`, the original document ID and metadata, and `status: already_indexed` without
re-parsing or re-embedding. The first upload returns `201` and `status: indexed`. A
concurrent retry that finds `indexing` returns `409` with the document ID rather than
claiming incomplete retrieval state is ready. The database uniqueness guard resolves
races between the pre-check and insert. If vector indexing or final publication fails,
the existing compensating flow removes vectors before incomplete metadata.

## Why this approach

- The digest is derived from the payload, so clients do not need to coordinate keys.
- Retries reuse both stored chunks and embeddings, reducing cost and duplicate search
  evidence.
- Database uniqueness is effective across processes and API replicas.
- Explicit publication state distinguishes “metadata exists” from “safe to retrieve.”
- The response remains transparent: clients can distinguish creation from replay.

## Alternatives considered

- Client-supplied idempotency keys: useful for logical operations, but separate clients
  can still upload the same bytes under different keys.
- Filename-based identity: rejected because filenames are mutable and non-unique.
- Return the digest as the public document ID: rejected to avoid changing the existing
  opaque-ID API and exposing a stable cross-system content identifier.
- Allow duplicate documents and deduplicate search results: rejected because it still
  pays parsing, embedding, storage, and indexing costs.
- Hold a database lock through parsing and remote embedding: rejected because long
  transactions consume connections and amplify provider latency.

## Tradeoffs / risks

- Byte-identical content is one logical document even when uploaded under a new name;
  the first filename remains canonical.
- Semantically identical files with different bytes are not deduplicated.
- A crashed process can leave an `indexing` row. Production should add a bounded-age
  reconciliation job; operators can currently inspect the returned ID and delete it.
- Adding the unique index requires existing deployments to remove historical duplicate
  digests before upgrading if they already contain them.

## How it fits the architecture

The API computes the digest after size/type validation and consults the existing
`DocumentRepository` boundary. Parsing and embedding stay provider-independent. The
repository persists citation chunks in the indexing state, `VectorStore` publishes the
retrieval data, and the repository then marks the source indexed. Memory adapters model
the same states for local development; PostgreSQL provides the cross-replica uniqueness
guarantee. Bounded metrics distinguish new, deduplicated, in-progress, and failed
lookups without exposing digests or document IDs.

## Interview explanation

“Retries used to create duplicate vectors and citations. I made ingestion
content-addressed: SHA-256 is checked before expensive work and protected by a unique
database index for cross-replica races. Metadata has an indexing state and is published
only after vectors are durable, so a racing retry never receives a false success. This
is a small state machine with compensating deletion across two storage boundaries. The
main tradeoff is that exact bytes share one identity; a production follow-up would
reconcile stale indexing claims after worker crashes.”
