# ADR-003: Synchronous local ingestion and retrieval orchestration

## Context / Problem

Parsing, chunking, embedding, and vector retrieval existed as independently tested
components, but the HTTP upload endpoint stopped after validation. That meant the
repository did not yet expose an exerciseable retrieval path and source offsets were
not carried through an API response.

## Decision

For the local development path, make text and Markdown upload synchronously execute the
pipeline: validate bytes → parse normalized UTF-8 text → create overlapping chunks →
embed chunks → replace the document in the vector store. Add a POST `/search` endpoint
that embeds the query and returns ranked source chunks including document ID, chunk
index, text, and exact normalized-text character offsets.

Use one application-scoped embedder and vector store so indexed documents remain
searchable for the life of the process. Add `/ready` validation that the embedding
provider and vector store use the same dimensionality. Recognize PDF as an intended
format but return HTTP 501 until a real extraction implementation exists instead of
pretending raw PDF bytes are searchable text.

## Why this approach

Connecting the existing boundaries proves the complete retrieval control flow without
erasing those boundaries. Synchronous processing is transparent and deterministic for
small local documents, and the source span returned by search establishes the
provenance contract needed by citation-backed answer generation. Explicit PDF rejection
keeps API behavior honest while leaving room for a later extractor.

## Alternatives considered

- Keep upload as acceptance-only: preserves staged development but leaves the core
  retrieval path disconnected.
- Put parsing, embedding, and cosine logic directly in the endpoint: fewer abstractions,
  but would undo the existing provider/store boundaries and make production adapters
  harder to introduce.
- Start with a background queue immediately: more production-like for large documents,
  but adds job state and infrastructure before the synchronous correctness path is
  connected.
- Treat PDF bytes as UTF-8 or extract strings heuristically: easy to demo but unreliable
  and misleading for real PDF structure.

## Tradeoffs / Risks

The in-memory vector store is process-local and synchronous ingestion blocks the request,
so this path is not appropriate for large documents or multi-instance deployments.
The hashing embedder is lexical rather than semantic. Uploaded document metadata is not
yet durable, and PDF extraction is intentionally unavailable. Later work should move
metadata/chunks to PostgreSQL/pgvector, add a semantic embedding provider, and introduce
asynchronous ingestion once durable job state exists.

## How it fits the architecture

HTTP upload orchestrates the existing parser, chunker, embedder port, and vector-store
port. Search uses the same embedder and vector store, returning the original `TextChunk`
provenance fields. The next answer-generation layer can consume those search results and
produce citations without knowing which embedding or vector-storage adapter is active.

## Interview explanation

"I first built parsing, embeddings, and vector storage behind separate contracts, then
connected them with a synchronous local ingestion path. Upload now produces traceable
chunks, indexes them, and search returns exact source offsets. I deliberately return 501
for PDFs until a real extractor exists rather than claiming support I don't have. The
synchronous in-memory path establishes correctness; the production evolution is durable
Postgres/pgvector storage plus asynchronous ingestion and a semantic embedding provider."
