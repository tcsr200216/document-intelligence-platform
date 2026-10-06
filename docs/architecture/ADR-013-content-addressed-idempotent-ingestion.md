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
claiming incomplete retrieval state is ready.

Every indexing row also carries a bounded lease. An identical retry may atomically
claim the row only when its lease is absent or expired, then rebuild vectors from the
persisted citation chunks and publish the same document ID. A successful recovery
returns HTTP `200` and `status: recovered_indexed`. The lease timestamp is also a
fencing token: final publication succeeds only if the worker still owns that exact
lease. A failed embedding call releases only its own lease so a retry can proceed;
an expired worker cannot release or publish a newer owner's work. If it discovers the
lost lease at publication, it returns a conflict without compensating metadata or
vectors that the replacement worker now owns.

The database uniqueness guard resolves races between the pre-check and insert. If
vector indexing or final publication fails, the existing compensating flow removes
vectors before incomplete metadata.

## Why this approach

- The digest is derived from the payload, so clients do not need to coordinate keys.
- Retries reuse both stored chunks and embeddings, reducing cost and duplicate search
  evidence.
- Database uniqueness is effective across processes and API replicas.
- Explicit publication state distinguishes “metadata exists” from “safe to retrieve.”
- A short compare-and-swap lease recovers from process crashes without a background
  queue, long database transaction, or replica-local lock.
- Exact-token publication fences stale workers after another replica takes ownership.
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
- Unconditionally retry any `indexing` row: rejected because active requests would
  duplicate provider work and race publication.
- Periodic reconciliation worker: useful at larger scale, but unnecessary for recovery
  while upload retries can safely reclaim the persisted work.

## Tradeoffs / risks

- Byte-identical content is one logical document even when uploaded under a new name;
  the first filename remains canonical.
- Semantically identical files with different bytes are not deduplicated.
- Recovery is retry-driven: a stale row remains `indexing` until the same content is
  uploaded again. A future worker can proactively scan stale rows without changing the
  claim protocol.
- The lease must exceed normal vector publication time. An unusually slow worker can
  lose ownership and receive a publication error, while the replacement safely wins.
- Adding the unique index requires existing deployments to remove historical duplicate
  digests before upgrading if they already contain them.

## How it fits the architecture

The API computes the digest after size/type validation and consults the existing
`DocumentRepository` boundary. Parsing and embedding stay provider-independent. The
repository persists citation chunks in the indexing state, `VectorStore` publishes the
retrieval data, and the repository then marks the source indexed. Memory adapters model
the same states for local development; PostgreSQL provides the cross-replica uniqueness
guarantee and performs lease claims as conditional updates. Recovery deliberately
reuses persisted chunks but recomputes embeddings behind the existing provider boundary;
vector replacement remains idempotent for the stable document ID. Bounded metrics
distinguish new, deduplicated, recovered, in-progress, and failed lookups without
exposing digests or document IDs.

## Interview explanation

“Retries used to create duplicate vectors and citations. I made ingestion
content-addressed: SHA-256 is checked before expensive work and protected by a unique
database index for cross-replica races. Metadata has an indexing state and is published
only after vectors are durable, so a racing retry never receives a false success. This
is a small state machine with compensating deletion across two storage boundaries. To
handle crashes, indexing rows carry expiring leases claimed with a conditional update;
the timestamp also fences stale workers at publication. Recovery reuses durable chunks
and the same document ID instead of parsing again. The main tradeoffs are retry-driven
recovery and choosing a lease long enough for normal indexing latency.”
