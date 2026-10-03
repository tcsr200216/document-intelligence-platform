# ADR-006: Explicit semantic embedding provider selection

## Context / Problem

The deterministic hashing embedder makes local development and tests reproducible,
but it measures token overlap rather than semantic similarity. The platform needs a
real semantic provider without coupling ingestion and retrieval to one vendor or
requiring an API key for every developer and CI run.

## Decision

Keep the existing `Embedder` boundary and add an OpenAI-compatible adapter selected
with `EMBEDDING_PROVIDER`. `hashing` remains the default. Selecting `openai` requires
an environment-supplied key and supports configurable model, dimensions, base URL,
and timeout.

The remote adapter sends batches, restores provider results to input order, and
validates response count, unique indices, vector dimensions, numeric values, and
finiteness before returning any vector. Provider and model version are exposed by
the readiness response. A remote error is explicit; the service never silently
falls back to hashing after a model has been selected.

## Why this approach

- The application orchestration and vector-store code continue to depend on a
  small provider protocol rather than an SDK-specific type.
- OpenAI-compatible HTTP keeps the adapter testable with an in-process mock and
  permits compatible gateways through configuration.
- The hashing default preserves deterministic, secret-free local and CI behavior.
- Strict batch validation prevents incorrectly ordered or malformed vectors from
  entering the index.

## Alternatives considered

- **Vendor SDK directly in API handlers:** less adapter code, but couples business
  flow to an SDK and makes deterministic tests harder.
- **Automatic fallback on remote failure:** improves apparent availability but can
  mix incompatible vector spaces and silently degrade retrieval correctness.
- **Run a local transformer model by default:** provides semantics without a remote
  API, but substantially increases image size, startup time, and hardware needs.

## Tradeoffs / risks

Remote semantic embedding adds latency, cost, rate limits, and an external
availability dependency. The API currently performs embedding synchronously. A
provider/model/dimension change requires an empty or rebuilt vector index; the
current in-memory store naturally starts empty on restart, while a future durable
store must enforce model-version compatibility. The API key must come from runtime
secret management and is never returned in errors or readiness data.

## How it fits the architecture

Upload and search call the same `Embedder` protocol as before. The vector store is
constructed from the selected provider's declared dimensions, so its existing
validation protects both ingestion and query paths. Repository and citation-offset
boundaries are unchanged.

## Interview explanation

“I kept a deterministic hashing model for local reproducibility, then added a real
semantic provider behind the same interface. I deliberately made provider selection
explicit and validated ordered batch responses because silent fallback or mixed
embedding models would put vectors from incompatible spaces into the same index and
make cosine similarity meaningless.”
