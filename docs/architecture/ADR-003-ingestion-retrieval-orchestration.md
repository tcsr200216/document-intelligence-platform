# ADR-003: Synchronous ingestion, retrieval, and document lifecycle orchestration

## Context / Problem

Parsing, chunking, embedding, metadata persistence, vector indexing, retrieval, and
answering use separate boundaries. The API must connect them without allowing searchable
chunks to outlive their citation source. It must also handle partial failures across the
metadata repository and vector store, which do not share one transaction, and give users
a supported way to remove an indexed document.

## Decision

Keep ingestion synchronous for the current 10 MB upload limit. The upload sequence is:
validate → parse → chunk → embed → persist metadata/chunks → replace vectors. Publishing
vectors last guarantees that every searchable chunk has a persisted citation source. If
vector indexing fails, the orchestrator compensates by deleting the metadata and chunks
saved earlier and records the rollback outcome.

Expose `DELETE /documents/{document_id}`. It confirms the document exists, idempotently
removes all model-scoped vectors first, then deletes metadata and citation chunks in one
repository transaction. Removing vectors first favors retrieval safety: a partial deletion
may temporarily retain non-searchable metadata, but never searchable evidence whose source
record has already disappeared. Retrying completes that state safely. Both in-memory and
PostgreSQL/pgvector adapters implement the same delete contracts.

## Why this approach

The ordering preserves the citation invariant without coupling HTTP handlers to SQL tables.
Compensation handles the common index-failure path while keeping the existing provider and
storage boundaries. Vector-first deletion makes privacy and citation correctness safer than
metadata-first deletion, and idempotent adapter operations make retries predictable.

## Alternatives considered

- **One cross-store database transaction:** strongest atomicity, but would merge the
  repository and vector-store boundaries and prevent independent adapters.
- **Metadata-first deletion:** simpler, but a vector-delete failure would leave searchable
  chunks with no durable citation record.
- **Soft deletion only:** useful for audit workflows, but does not satisfy actual data removal
  and still requires retrieval filtering everywhere.
- **Background ingestion queue:** appropriate for larger documents and higher throughput,
  but requires durable job state, workers, retries, and reconciliation. It is not justified
  solely to process the current bounded synchronous workload.
- **Ignore partial failures:** leaves orphaned records and makes operational recovery unclear.

## Tradeoffs / Risks

Compensation is not equivalent to a distributed transaction: a process crash or a second
storage failure can still leave non-searchable metadata requiring reconciliation. A metadata
failure after vector deletion returns a retryable error and deliberately keeps the safer
non-searchable state. Concurrent delete and read requests are not serialized, so a request
already holding retrieved context may finish while deletion runs. Synchronous parsing and
embedding still occupy the request until ingestion completes.

## How it fits the architecture

The FastAPI layer owns workflow ordering while parser, embedder, repository, vector store,
hybrid retriever, and answer generator remain separate ports. The repository owns atomic
metadata/chunk deletion; the vector store owns model-scoped index deletion. Bounded metrics
record ingestion rollback and document deletion outcomes without labeling document IDs.

## Interview explanation

“My metadata repository and vector index are intentionally separate boundaries, so I made
the orchestration order enforce the important invariant: vectors are published only after
their citation source is durable. If indexing fails, I compensate by removing the earlier
metadata write. Deletion reverses that safety priority—vectors go first, then metadata—so a
partial failure can leave harmless metadata but never searchable orphan evidence. The
operations are idempotent and retryable, and I explicitly document that this is a saga-like
compensation rather than pretending it is a distributed transaction.”
