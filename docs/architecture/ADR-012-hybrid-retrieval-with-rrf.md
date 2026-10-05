# ADR-012: Hybrid retrieval with Reciprocal Rank Fusion

## Context / Problem

Vector similarity retrieves conceptual matches, but it can under-rank exact evidence such as
policy codes, product identifiers, names, and short quoted phrases. Pure keyword search has the
opposite weakness: it misses useful paraphrases. Citation-backed answers need both forms of
evidence while preserving the existing document, chunk, character-offset, and page provenance.

## Decision

Search and question answering use a `HybridRetriever` that requests independent semantic and
lexical candidate lists from the `VectorStore` boundary and combines them with weighted
Reciprocal Rank Fusion (RRF). Semantic rank has weight `0.65`, lexical rank has weight `0.35`,
the RRF constant is `60`, and each backend contributes up to four times the requested result
count. Results are deduplicated by document ID and chunk index, deterministically tie-broken,
and returned with a fusion score normalized against the theoretical first-place score.

The in-memory adapter uses deterministic Unicode token matching. PostgreSQL uses
`websearch_to_tsquery` against a stored generated `tsvector` column with a GIN index; semantic
retrieval remains pgvector cosine distance. Both paths return the original `TextChunk`, so
citation offsets and page ranges are unchanged.

## Why this approach

RRF combines ranks rather than incomparable cosine and full-text scores. It is deterministic,
does not require a labeled training set, and keeps ranking policy outside storage adapters.
Fetching a bounded candidate pool makes fusion inexpensive while allowing a strong result from
either retriever to enter the final list.

## Alternatives considered

- **Vector search only:** simpler, but unreliable for exact identifiers and literal evidence.
- **Keyword search only:** strong on exact terms, but weak on paraphrases and meaning.
- **Weighted raw-score addition:** rejected because cosine similarity and PostgreSQL text rank
  have different scales and calibration.
- **Learned reranker:** potentially stronger, but adds latency, cost, model operations, and a
  need for representative relevance judgments that this project does not yet have.

## Tradeoffs / risks

Hybrid retrieval performs two bounded searches per request and needs an additional PostgreSQL
index. English text-search configuration can stem English terms well but is not yet multilingual.
Fixed fusion weights are transparent and testable, but not workload-tuned. A normalized RRF score
is useful for stable ordering and the existing answer threshold, but it is not a probability and
must not be interpreted as one.

## How it fits the architecture

Embedding providers remain responsible only for vectors. `VectorStore` adapters expose semantic
and lexical candidate retrieval, while `HybridRetriever` owns cross-retriever policy. The FastAPI
search and question endpoints share this component, and answer generators continue to receive
the same provenance-bearing `SearchHit` objects.

## Interview explanation

“Semantic search is good at meaning but can miss exact codes and phrases, while keyword search
has the reverse tradeoff. I kept both behind the storage boundary and fused their ranks with RRF,
because the raw scores are not directly comparable. PostgreSQL gives each path a real index—HNSW
for pgvector and GIN for full text—and the fusion layer preserves exact citation offsets. The
choice is deterministic and testable today, with a clear path to tune weights or add a reranker
after collecting real relevance judgments.”
