# Document Intelligence Platform

A Python/FastAPI document ingestion and retrieval service. The currently implemented
path is **upload → UTF-8 text/Markdown parsing → deterministic overlapping chunks →
local embeddings → indexed cosine search**. Chunk text and source offsets are also
stored in a repository adapter (in memory or PostgreSQL). This is an evolving
portfolio project: a true semantic provider, durable vector backend, PDF extraction,
and grounded answer generation are future milestones, **not current capabilities**.

## Architecture

```text
HTTP upload -> validate -> parse text -> chunk with exact source offsets
                                    -> HashingEmbedder -> InMemoryVectorStore
                                    -> DocumentRepository (memory or PostgreSQL)
HTTP search -> embed query -> rank chunks -> return document, chunk, score and source offsets
```

See `docs/architecture/` for the recorded decisions and tradeoffs. The current
hashing embedder measures deterministic token overlap, not semantic similarity.
With PostgreSQL configured, document metadata and chunk provenance survive restarts,
but the *vector index does not*: re-upload is required for search until a durable
vector adapter or index rebuild path is added.

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
The smoke script uploads `data/sample_document.txt`, fetches its metadata, and
performs a document-scoped search. It exits nonzero if any step fails.

### Run with Docker and PostgreSQL

Requires Docker Engine/Desktop with Compose:

```bash
docker compose up --build -d
docker compose ps
```

The API is at http://localhost:8000/docs and PostgreSQL is internal to Compose.
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

Supported formats are `.txt` and `.md` with UTF-8 text (up to 10 MB).
PDF upload returns an explicit 501 until extraction is implemented.

## Tests and CI

`python -m pytest -q` covers parsing, chunking, embeddings, source provenance,
vector retrieval, repository storage, and HTTP ingestion/search. The GitHub Actions
workflow also compiles Python sources, runs critical lint checks and builds the
container image. CI passing does not prove cloud deployment or PDF/semantic Q&A support.

## Deploy

The provided `Dockerfile` and `compose.yaml` support a single-host container
deployment (for example on a Linux VM). Set a **private production PostgreSQL**
instance and supply `DATABASE_URL` securely to the container; do not publish
PostgreSQL to the Internet or reuse the Compose demo password. Configure HTTPS
and authentication at a trusted reverse proxy, set resource/upload limits,
and monitor `/health` and `/ready`. Deploy the image with one API replica for
the current in-memory vector index, which is not shared across replicas.

For a test deployment on a VM, install Docker/Compose, clone this repository,
replace the development database credentials, and run `docker compose up --build -d`.
Then use the smoke test against the mapped HTTP endpoint (prefer an authenticated
HTTPS reverse proxy for remote access). Consult `docs/architecture/` before
scaling: vector index durability, migrations, auth and PDF/Q&A are outstanding.
