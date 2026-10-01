# ADR-004: Durable document metadata and citation-span repository

## Context / Problem
The ingestion flow indexes chunks in memory but previously discarded document metadata outside the upload response. Citation-backed Q&A needs a durable source of truth for document identity, checksum, and exact normalized chunk spans. Local development must remain runnable without PostgreSQL, and vector persistence is not implemented yet.

## Decision
Introduce a `DocumentRepository` port with in-memory and SQLAlchemy adapters. Persist document metadata and every normalized chunk's index, text, `start_char`, and `end_char`. The SQL adapter stores a document and its complete chunk set in one transaction and is selected when `DATABASE_URL` is configured; memory remains the zero-setup fallback. Ingestion persists the citation source before publishing vectors to the in-memory retriever. Add `GET /documents/{document_id}` and include repository health in readiness.

This intentionally does not claim durable vector retrieval. Vectors remain process-local until a pgvector-backed `VectorStore` exists.

## Why this approach
Separating metadata persistence from vector search keeps source-of-truth records and retrieval infrastructure independently replaceable. Exact normalized spans are stored, so later answers can resolve a hit back to the same boundaries used during indexing. One SQL transaction prevents partially replaced chunk sets.

## Alternatives considered
- Persist nothing until pgvector exists: simpler, but loses provenance across restarts.
- Put metadata only in the vector store: couples source records to retrieval technology.
- Require PostgreSQL everywhere: production-like but harms local onboarding.
- Recreate chunks from whole text later: risks citation drift when chunking changes.

## Tradeoffs / Risks
PostgreSQL metadata can survive restart while the current in-memory vector index cannot, so old documents are inspectable but not searchable after restart until durable vector indexing/rebuild exists. `create_all` is early bootstrap behavior and should become versioned migrations as the schema evolves. Full chunk text duplicates content in exchange for stable citation provenance.

## How it fits the architecture
Upload → parse → deterministic chunks → embeddings → persist document/chunk provenance → publish vectors to retrieval. The document repository owns durable source metadata; the vector-store boundary owns similarity search.

## Interview explanation
"I separated document provenance from vector retrieval. PostgreSQL stores the checksum and exact normalized chunk text and offsets transactionally, while a repository interface preserves a no-infrastructure local mode. That gives citations a durable source of truth without pretending the in-memory vector index is durable. The next step is pgvector behind the existing VectorStore boundary."
