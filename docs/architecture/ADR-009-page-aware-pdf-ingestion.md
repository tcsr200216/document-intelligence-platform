# ADR-009: Page-aware PDF ingestion

## Context / Problem

The upload contract accepted `.pdf` but returned `501`, leaving a major document
intelligence use case disconnected from parsing, retrieval, and cited Q&A. PDF text
also needs page provenance: character offsets alone are hard for a user to locate in
the original file. Treating the entire extracted PDF as one stream could create chunks
that cross page boundaries and make citations ambiguous.

## Decision

Use `pypdf` to extract embedded text locally. Normalize each readable page with the
same line-ending and trailing-whitespace rules as text uploads, join readable pages
with two newlines, and record each page's exact offsets in that normalized document.
Chunk each PDF page independently, preserving one-based `page_start` and `page_end`
alongside global normalized character offsets through PostgreSQL, pgvector, search,
and Q&A citations.

Reject malformed, password-protected, and textless/image-only PDFs with a clear `422`.
Do not silently invoke OCR or send documents to a remote provider. Add nullable page
columns so existing text records and existing PostgreSQL data remain compatible; the
service applies the additive columns idempotently at startup until managed migrations
are introduced.

## Why this approach

Local extraction keeps the key-free development path reproducible and avoids sending
potentially sensitive documents to a third party. Page-bounded chunks make every PDF
citation directly locatable while global offsets retain the same invariant used by
text documents. Nullable provenance fields avoid inventing page numbers for formats
that do not have pages.

## Alternatives considered

- **Cloud document AI:** stronger layout and OCR support, but introduces cost, keys,
  data-governance concerns, and provider coupling.
- **OCR every PDF:** supports scans but is operationally heavier and can alter text in
  ways that weaken exact citation traceability.
- **Chunk the combined PDF text:** simpler, but chunks can straddle pages and produce
  unclear citations.
- **Return page numbers without offsets:** user-friendly but discards the existing
  machine-verifiable source-span contract.

## Tradeoffs / risks

`pypdf` extracts embedded text but does not preserve complex visual reading order and
does not OCR images. Normalization means offsets refer to the stored normalized text,
not PDF byte positions. Startup DDL is intentionally limited to additive nullable
columns; a production deployment should replace it with reviewed schema migrations.
Very large PDFs remain protected by the existing 10 MB upload limit, but parsing is
still synchronous and should move to a bounded worker queue at higher traffic.

## How it fits the architecture

The parser boundary now dispatches by extension and produces a common normalized
document model. The chunker owns page boundaries, while repository and vector-store
adapters persist the same provenance fields. Retrieval and answer generation remain
format-agnostic and simply return the stored source contract.

## Interview explanation

"I added real PDF ingestion without weakening citation traceability. Each readable
page is normalized and chunked independently, then its one-based page number and exact
normalized offsets travel through metadata storage, pgvector retrieval, and the Q&A
response. I deliberately fail clearly on scans instead of pretending OCR quality, and
kept extraction local to preserve a key-free, privacy-conscious development path."
