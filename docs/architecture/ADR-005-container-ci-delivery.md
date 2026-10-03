# ADR-005: Reproducible container and CI delivery

## Context / Problem

The ingestion/retrieval service now has API and repository tests, but a new developer
cannot verify the system from the placeholder README. A deployment must offer
reproducible dependencies and a working PostgreSQL option while retaining a no-service
development path.

## Decision

Package the `app` Python module via setuptools; add a Python 3.12 Dockerfile, Compose
topology with a health-checked PostgreSQL dependency, explicit sample document and an
HTTP smoke test. GitHub Actions installs development dependencies, compiles modules,
runs critical lint/test checks and builds the container. Document both a local Python
workflow and a single-host Compose deployment, including current vector persistence
limitations.

## Why this approach

Tests prove individual behavior; a smoke test checks that the public HTTP path is
connected. A single image is easy to reproduce locally or on a container host. Compose
exercises SQL integration without forcing developers to install PostgreSQL manually.

## Alternatives considered

- Require a managed cloud service for every demo: introduces credentials/cost.
- Ship only a Docker image and no local instructions: obscures debugging and onboarding.
- Add Kubernetes immediately: premature complexity for the current single-process
  in-memory vector index.

## Tradeoffs / Risks

The Compose password and direct port mapping are for **local development only**.
The vector index is still process-local and is not reconstructed from persisted chunk
metadata; multiple API replicas would return inconsistent search results. Build/pytest
CI does not establish that every deployment environment is production-ready. A future
hosted rollout must add auth, secret management, migrations, durable vectors and
production telemetry.

## How it fits the architecture

HTTP API and its storage/embedding abstractions run unchanged in Python or inside the
container. `DATABASE_URL` selects the repository adapter; the sample upload/search
smoke path verifies the API contract; CI prevents basic regressions before delivery.

## Interview explanation

"I packaged the API into a reproducible container with a PostgreSQL Compose topology
and a no-dependency local mode. I added a sample HTTP smoke test alongside unit and
integration tests so setup can be checked end to end. I documented that durable SQL
metadata does not yet make the in-memory vector index durable, rather than claiming
that a multi-replica deployment is already safe."
