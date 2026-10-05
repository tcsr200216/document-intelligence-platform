# Document Intelligence Platform

A Python/FastAPI document ingestion and retrieval service. The currently implemented
path is **upload → text/Markdown or page-aware PDF parsing → deterministic chunks →
pluggable embeddings → indexed cosine search**. Chunk text, source offsets, and
vectors are stored behind repository adapters (in memory or PostgreSQL/pgvector).
This is an evolving portfolio project: OpenAI-compatible semantic embeddings are available through
environment configuration, while the deterministic hashing provider remains the
key-free local default. Citation-backed Q&A is available through a deterministic
extractive baseline or an OpenAI-compatible grounded generator. Text-based PDFs are
extracted locally; scanned/image-only PDFs fail clearly because OCR is not enabled.

## Architecture

```text
HTTP upload -> validate -> parse text/PDF -> page-bounded chunks with exact offsets
                                    -> Embedder -> VectorStore (memory or pgvector)
                                    -> DocumentRepository (memory or PostgreSQL)
HTTP search -> embed query -> rank chunks -> return document, chunk, score and source offsets
HTTP question -> retrieve chunks -> grounded answer -> return only validated source citations
```

See `docs/architecture/` for the recorded decisions and tradeoffs. The current
hashing embedder measures deterministic token overlap, not semantic similarity.
The OpenAI adapter performs real remote semantic embedding with strict response
dimension, index, and numeric validation. Provider/model details appear in `/ready`.
With PostgreSQL configured, document metadata, chunk provenance, and model-scoped
vectors survive restarts. The pgvector adapter performs database-side cosine search
through an HNSW index for vectors up to 2,000 dimensions (and an exact database
scan above that limit). It refuses startup if the configured embedding model or
dimensions differ from the durable index contract.

Search and Q&A share a model-scoped query embedding cache. Local runs use a finite-TTL
in-memory adapter; setting `REDIS_URL` enables a shared Redis adapter for multiple API
replicas. Cache keys include the embedding model contract and a SHA-256 query digest,
never raw question text. Cached vectors are validated before use, and request handling
falls back to live embedding when Redis has a transient failure. See
[ADR-011](docs/architecture/ADR-011-model-scoped-query-embedding-cache.md).

Every response includes a generated `X-Request-ID`, and the service emits one
structured completion log per request. Prometheus metrics at `/metrics` cover HTTP
traffic and latency, ingestion stages and duration, chunk counts, retrieval outcomes,
result counts, cache outcomes, and grounded-answer outcomes. Labels use route templates, document
format, operation, provider, and finite outcome values; filenames, document IDs,
questions, chunk text, and exception messages are never labels.

## Run locally

Requires **Python 3.12+**. With no `DATABASE_URL`, no external services are needed:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
uvicorn app.main:app --reload
```

In a second terminal (same virtual environment):

```bash
python scripts/smoke_test.py
python -m pytest -q
```

Open http://localhost:8000/docs for interactive API documentation.
The smoke script builds and uploads a real two-page text PDF, fetches its metadata,
and exercises document-scoped search and citation-backed Q&A. It exits nonzero if
any step fails.

### Use semantic embeddings

Hashing is the default so local runs and CI need no secret. To use an
OpenAI-compatible semantic provider, set the following in `.env` before starting
the API:

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_DIMENSIONS=1536
OPENAI_API_KEY=your-key-from-a-secret-manager
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

`OPENAI_BASE_URL` can point to a compatible gateway. The service fails at startup
if `openai` is selected without a key, and remote errors fail the request rather
than silently switching models. Never mix embeddings from different models in one
index; restart with an empty/rebuilt vector index after changing provider, model,
or dimensions. See [ADR-006](docs/architecture/ADR-006-semantic-embedding-provider.md).

### Ask citation-backed questions

The default `ANSWER_PROVIDER=extractive` quotes the strongest relevant chunk and
needs no key. Set `ANSWER_PROVIDER=openai` plus `OPENAI_API_KEY` to synthesize a
grounded answer through an OpenAI-compatible chat endpoint. Remote responses are
accepted only when every cited source number maps to retrieved context; invalid or
uncited answers fail closed.

```bash
curl -fsS -X POST http://localhost:8000/questions \
  -H "Content-Type: application/json" \
  -d '{"question":"Where is metadata stored?","document_id":"YOUR_DOCUMENT_ID"}'
```

The response includes answer status, provider, and citations containing the exact
document ID, chunk index, chunk text, cosine score, normalized source character
offsets, and PDF page range when applicable.
When context is absent or below the configured relevance threshold, the service
returns `insufficient_context` with no citations. See
[ADR-007](docs/architecture/ADR-007-citation-backed-question-answering.md).

### Run with Docker and PostgreSQL

Requires Docker Engine/Desktop with Compose:

```bash
docker compose up --build -d
docker compose ps
```

The API is at http://localhost:8000/docs; pgvector-enabled PostgreSQL and Redis are
internal to Compose. `/ready` verifies both durable storage and the configured cache.
The credentials in `compose.yaml` are **local development examples only**.
Run `python scripts/smoke_test.py` on the host after installing the Python project,
or use the following equivalent health check:

```bash
curl -fsS http://localhost:8000/ready
```

Use `docker compose down` to stop services; `docker compose down -v` also deletes
the local database volume. Never use example credentials for a public deployment.

### Example API workflow

```bash
curl -fsS -F "file=@data/sample_document.txt;type=text/plain" \
  http://localhost:8000/documents/upload
# From the JSON response, copy document_id:
curl -fsS http://localhost:8000/documents/YOUR_DOCUMENT_ID
curl -fsS -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Where is metadata stored?","document_id":"YOUR_DOCUMENT_ID","limit":3}'
```

Supported formats are UTF-8 `.txt`/`.md` and text-based `.pdf` files up to 10 MB.
PDF chunks never cross page boundaries, and citations include one-based page numbers.
The parser does not perform OCR; scanned/image-only and password-protected PDFs return
an explicit validation error instead of being indexed as empty content. See
[ADR-009](docs/architecture/ADR-009-page-aware-pdf-ingestion.md).

## Tests and CI

`python -m pytest -q` covers parsing, chunking, embeddings, source provenance,
vector retrieval, repository storage, and HTTP ingestion/search. The GitHub Actions
workflow compiles Python sources, runs lint and tests, builds the image, launches the
API with PostgreSQL/pgvector, and runs the complete PDF upload → persistence → retrieval
→ cited Q&A flow before and after an API restart. CI passing does not prove cloud
deployment or OCR support.

## Deploy

The provided `Dockerfile` and `compose.yaml` support a single-host container
deployment (for example on a Linux VM). Set a **private production PostgreSQL**
instance with the pgvector extension and supply `DATABASE_URL` securely to the
container. Supply a private Redis instance through `REDIS_URL` when running multiple
API replicas; do not publish PostgreSQL or Redis to the Internet or reuse the Compose
demo password. Configure HTTPS and authentication at a trusted reverse proxy, set
resource/upload limits, monitor `/health` and `/ready`, and scrape `/metrics` only
from a private monitoring network. PostgreSQL-backed vectors and Redis-cached query
embeddings are shared across API replicas; the no-database and no-Redis fallbacks
remain process-local and are intended for development.

For a test deployment on a VM, install Docker/Compose, clone this repository,
replace the development database credentials, and run `docker compose up --build -d`.
Then use the smoke test against the mapped HTTP endpoint (prefer an authenticated
HTTPS reverse proxy for remote access). Consult `docs/architecture/` before
scaling: managed migrations, authentication, OCR, and background ingestion remain
future production work.

## Operations and observability

Use `/health` for process liveness, `/ready` for dependency and embedding-contract
readiness, and `/metrics` for Prometheus scraping. A minimal alerting policy should
watch readiness failures, non-2xx request rates, ingestion stage errors, provider
errors, and latency percentiles. Correlate a client-visible `X-Request-ID` with the
JSON request-completion log when investigating a failure. Metrics deliberately avoid
raw paths and source identifiers, but the endpoint should still remain private because
operational traffic patterns may be sensitive. See
[ADR-010](docs/architecture/ADR-010-bounded-cardinality-observability.md).
