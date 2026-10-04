# ADR-007: Citation-backed question answering with validated grounding

## Context / Problem

Retrieval returns traceable chunks but leaves callers to construct an answer. A
question-answering layer must not detach claims from the exact retrieved evidence or
pretend to answer when the index has no relevant context. Local development also
needs a useful path without an external model key.

## Decision

Add `/questions` as retrieve-then-answer orchestration over the existing embedder and
vector-store boundaries. Every response includes a status, answer provider, and only
the citations selected by the answer generator. Citations preserve document ID,
chunk index, text, similarity score, and exact normalized-source character offsets.

Provide two `AnswerGenerator` adapters. `extractive` is the deterministic default: it
quotes the strongest chunk when its score meets a configurable threshold and otherwise
abstains. `openai` sends numbered retrieved sources to an OpenAI-compatible chat
endpoint and requires structured JSON. The adapter validates status, answer length,
citation types and ranges, and requires at least one valid citation for an answer.
Invalid or unavailable remote output fails closed with HTTP 503.

## Why this approach

- Retrieval and generation remain separate, testable boundaries.
- The extractive baseline is fully local, deterministic, and cannot invent text.
- Remote generation can improve synthesis while citations remain constrained to the
  retrieved source set.
- Explicit `insufficient_context` behavior makes abstention part of the API contract.
- Exact offsets retain the provenance established during ingestion.

## Alternatives considered

- **Return retrieval only:** maximally transparent but does not complete the requested
  question-answering workflow.
- **Always use an LLM:** improves fluency but requires secrets, adds latency/cost, and
  makes local and CI operation dependent on an external service.
- **Trust citations written as free-form text:** easy to prompt, but document IDs and
  offsets can be fabricated. Numeric indices are validated against actual hits.
- **Answer without a relevance gate:** creates plausible-looking output for unrelated
  questions and weakens the grounding guarantee.

## Tradeoffs / risks

The extractive answer may be verbose and is not a synthesized explanation. The remote
adapter validates citation references, but a cited chunk does not mathematically prove
every generated claim; evaluation and stronger entailment checks remain future work.
Cosine thresholds depend on the embedding model and corpus and must be calibrated.
Question answering remains synchronous, so remote latency is on the request path.

## How it fits the architecture

The endpoint embeds the question, queries `VectorStore`, and passes `SearchHit` values
to `AnswerGenerator`. It does not bypass ingestion, persistence, embedding, or retrieval
boundaries. The response maps validated citation indices back to the original hits, so
the transport layer never accepts provider-invented document identifiers or offsets.

## Interview explanation

“I completed the RAG flow with a provider boundary for answering. Local mode quotes the
best source and abstains below a threshold; remote mode can synthesize, but it returns
only numeric citation indices that I validate against the retrieved chunks. The API
then builds citations from trusted search results, not model-generated metadata, so a
model cannot fabricate a document ID or character offset.”
