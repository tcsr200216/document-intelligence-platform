# ADR-010: Bounded-cardinality observability

## Context / Problem

The upload, retrieval, and citation-backed Q&A flow can fail at several distinct
boundaries: parsing, embedding, metadata persistence, vector indexing, retrieval, and
answer generation. Liveness and readiness probes show whether the process can serve,
but they do not explain workload health or help correlate a user-visible failure with
service logs. Naively labeling metrics with filenames, document IDs, questions, or
raw URLs would leak source identity and create unbounded Prometheus series.

## Decision

Expose Prometheus metrics at `/metrics`, add a server-generated `X-Request-ID` to
responses, and write a structured completion log for every HTTP request. Measure HTTP
counts and latency, ingestion stages and duration, chunk counts, retrieval outcomes
and result counts, and grounded-answer outcomes. Metric labels are restricted to
finite values: HTTP method, route template, status class, supported document format,
pipeline stage, operation, configured provider, and enumerated outcome.

Health, readiness, and metrics scrapes are excluded from HTTP workload metrics to
avoid probe traffic obscuring user traffic. `/metrics` is intended for a private
monitoring network even though its labels contain no source content or identifiers.

## Why this approach

Prometheus provides portable infrastructure-independent telemetry and works with the
existing container deployment. Route templates keep `/documents/{document_id}` to one
time series instead of one per document. Generated request IDs let clients report a
safe correlation token without trusting spoofable caller-provided IDs. Pipeline-stage
metrics identify the failing boundary without logging document contents or remote
provider responses.

## Alternatives considered

- **Logs only:** simple, but inefficient for rate, latency, and alert calculations.
- **OpenTelemetry tracing immediately:** richer distributed context, but adds exporter
  and collector complexity before this service has multiple deployed components.
- **Document IDs or filenames as labels:** convenient for debugging, but rejected due
  to privacy exposure and unbounded-cardinality cost.
- **Raw URL labels:** rejected because path parameters create a series per document.

## Tradeoffs / risks

Process-local counters reset when a replica restarts and must be aggregated by the
monitoring system. Histograms add modest memory and scrape overhead. Request IDs offer
correlation, not distributed tracing. A public metrics endpoint could reveal traffic
patterns, so deployment must restrict it at the network or reverse-proxy layer.

## How it fits the architecture

HTTP middleware observes transport-level behavior without coupling domain services to
FastAPI handlers. Explicit counters at orchestration boundaries describe parsing,
embedding, persistence, indexing, retrieval, and answer generation while the existing
provider and storage interfaces remain unchanged. This can later feed an
OpenTelemetry exporter without changing API contracts.

## Interview explanation

I instrumented the service around failure boundaries rather than attaching user or
document identity to telemetry. Prometheus gives operators rates and latency, while a
generated request ID connects a user report to structured logs. The critical design
choice was bounded cardinality: route templates and finite outcomes keep monitoring
cost predictable and avoid leaking document names, IDs, questions, or text.
