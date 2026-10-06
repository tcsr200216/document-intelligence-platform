# ADR-014: Keyset-paginated durable document inventory

## Context / Problem

After a process restart, users could fetch a document only when they had retained its
opaque ID. A durable system needs a way to discover indexed and in-progress documents.
Returning the entire corpus is unbounded, while offset pagination becomes slower at deep
pages and can skip or duplicate rows when new uploads arrive between requests.

## Decision

Add `GET /documents` behind the existing `DocumentRepository` boundary. Order rows by
`uploaded_at DESC, document_id DESC` and paginate with a keyset containing both values.
The API encodes that position, a schema version, and the optional lifecycle-status filter
into a URL-safe opaque cursor. A continuation request must use the same filter. Page size
is bounded to 100, and each adapter reads one extra row to determine whether another page
exists.

PostgreSQL and SQLite issue parameterized range predicates over the ordered keys and fetch
chunk counts separately for only the visible document IDs. The memory adapter implements
the same ordering and continuation contract. Cursor decoding strictly validates structure,
version, timestamp timezone, identifier length, and filter consistency.

## Why this approach

- Keyset pagination has stable cost and behavior as the document table grows.
- The document ID is a deterministic tie-breaker for equal upload timestamps.
- Filter-bound cursors prevent subtle cross-query continuation bugs.
- Opaque versioned tokens allow the wire representation to evolve without exposing SQL.
- The shared repository contract keeps API behavior identical in local and durable modes.

## Alternatives considered

- Return all documents: rejected because response size and database work are unbounded.
- Offset/limit pagination: simpler, but rejected because concurrent inserts shift offsets
  and deep pages require scanning and discarding preceding rows.
- Filename search only: rejected because filenames are neither unique nor sufficient for
  lifecycle inspection.
- Signed or encrypted cursors: unnecessary for this cursor because it grants no authority,
  contains no document content, and all decoded values are validated and parameterized.

## Tradeoffs / risks

- Clients cannot jump directly to an arbitrary page number.
- A client that loses its cursor must restart from the newest page.
- The token reveals its pagination fields if decoded; it is opaque by API contract, not a
  secret. Authorization must still be enforced independently before public deployment.
- Ordering is a snapshot-like traversal, not a transactionally frozen export; documents
  deleted between pages naturally disappear.

## How it fits the architecture

The endpoint reads only durable metadata and chunk counts through `DocumentRepository`;
it does not couple to the vector store or embedding provider. Upload publication states
from ADR-013 become operationally visible, while retrieval and citation flows remain
unchanged. The existing HTTP metrics observe the bounded route without adding document IDs
or cursor values as labels.

## Interview explanation

“The service used opaque document IDs but offered no way to rediscover them after restart.
I added a repository-level inventory using `(uploaded_at, document_id)` keyset pagination.
The ID breaks timestamp ties, the cursor is versioned and bound to its status filter, and
SQL fetches chunk counts only for the visible page. This avoids offset drift under
concurrent uploads and gives memory and PostgreSQL the same testable contract.”
