from __future__ import annotations

import json
import logging
import time
from uuid import uuid4

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger("document_intelligence.http")
logger.setLevel(logging.INFO)

HTTP_REQUESTS = Counter(
    "document_intelligence_http_requests_total",
    "Completed HTTP requests by bounded route and status class.",
    ("method", "route", "status_class"),
)
HTTP_REQUEST_DURATION = Histogram(
    "document_intelligence_http_request_duration_seconds",
    "HTTP request duration in seconds by bounded route.",
    ("method", "route"),
)
INGESTION_STAGES = Counter(
    "document_intelligence_ingestion_stages_total",
    "Document ingestion stage outcomes.",
    ("stage", "outcome", "format"),
)
INGESTION_DURATION = Histogram(
    "document_intelligence_ingestion_duration_seconds",
    "End-to-end successful ingestion duration in seconds.",
    ("format",),
)
INGESTED_CHUNKS = Histogram(
    "document_intelligence_ingested_chunks",
    "Chunks produced by successfully indexed documents.",
    ("format",),
    buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000),
)
RETRIEVAL_REQUESTS = Counter(
    "document_intelligence_retrieval_requests_total",
    "Retrieval operations by outcome.",
    ("operation", "outcome"),
)
RETRIEVAL_RESULTS = Histogram(
    "document_intelligence_retrieval_results",
    "Source chunks returned by successful retrieval operations.",
    ("operation",),
    buckets=(0, 1, 2, 3, 5, 10, 25, 50),
)
ANSWER_OUTCOMES = Counter(
    "document_intelligence_answer_outcomes_total",
    "Grounded-answer outcomes by provider.",
    ("provider", "status"),
)
EMBEDDING_CACHE_OPERATIONS = Counter(
    "document_intelligence_embedding_cache_operations_total",
    "Query embedding cache operations by outcome.",
    ("operation", "outcome"),
)
DOCUMENT_DELETIONS = Counter(
    "document_intelligence_document_deletions_total",
    "Document deletion outcomes across vector and metadata stores.",
    ("outcome",),
)
DOCUMENT_UPLOADS = Counter(
    "document_intelligence_document_uploads_total",
    "Document upload outcomes including content-addressed retries.",
    ("outcome", "format"),
)

_UNMEASURED_PATHS = frozenset({"/health", "/ready", "/metrics"})


class HttpObservabilityMiddleware(BaseHTTPMiddleware):
    """Add correlation IDs, structured completion logs, and bounded HTTP metrics."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            self._observe(request, request_id, status_code, started)
            raise

        response.headers["X-Request-ID"] = request_id
        self._observe(request, request_id, status_code, started)
        return response

    @staticmethod
    def _observe(request: Request, request_id: str, status_code: int, started: float) -> None:
        duration = time.perf_counter() - started
        route = getattr(request.scope.get("route"), "path", "unmatched")

        if request.url.path not in _UNMEASURED_PATHS:
            HTTP_REQUESTS.labels(
                request.method,
                route,
                f"{status_code // 100}xx",
            ).inc()
            HTTP_REQUEST_DURATION.labels(request.method, route).observe(duration)

        logger.info(
            json.dumps(
                {
                    "event": "http_request_completed",
                    "request_id": request_id,
                    "method": request.method,
                    "route": route,
                    "status": status_code,
                    "duration_ms": round(duration * 1000, 3),
                },
                separators=(",", ":"),
            )
        )


def metrics_response() -> Response:
    """Render the process registry in Prometheus text exposition format."""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
