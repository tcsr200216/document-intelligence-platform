import json
from uuid import UUID, uuid4

from conftest import build_text_pdf
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from app import main
from app.answering import ExtractiveAnswerGenerator
from app.cache import InMemoryEmbeddingCache
from app.embeddings import HashingEmbedder
from app.repository import InMemoryDocumentRepository
from app.vector_store import InMemoryVectorStore


def fresh_client(monkeypatch) -> TestClient:
    embedder = HashingEmbedder(dimensions=64)
    monkeypatch.setattr(main, "embedder", embedder)
    monkeypatch.setattr(
        main, "vector_store", InMemoryVectorStore(dimensions=embedder.dimensions)
    )
    monkeypatch.setattr(main, "document_repository", InMemoryDocumentRepository())
    monkeypatch.setattr(main, "embedding_cache", InMemoryEmbeddingCache())
    monkeypatch.setattr(
        main, "answer_generator", ExtractiveAnswerGenerator(min_relevance=0.0)
    )
    return TestClient(main.app)


def test_upload_indexes_persists_and_search_returns_traceable_source(monkeypatch) -> None:
    client = fresh_client(monkeypatch)
    payload = (
        b"PostgreSQL stores durable metadata.\n"
        b"Redis caches frequently requested values.\n"
        b"Kafka carries durable event streams."
    )

    upload = client.post(
        "/documents/upload",
        files={"file": ("architecture.txt", payload, "text/plain")},
    )

    assert upload.status_code == 201
    body = upload.json()
    assert body["status"] == "indexed"
    assert body["chunk_count"] == 1
    assert body["character_count"] == len(payload.decode())

    detail = client.get(f"/documents/{body['document_id']}")
    assert detail.status_code == 200
    assert detail.json()["filename"] == "architecture.txt"
    assert detail.json()["chunk_count"] == 1
    assert detail.json()["sha256"] == body["sha256"]

    search = client.post(
        "/search",
        json={"query": "Where does PostgreSQL store metadata?", "limit": 3},
    )
    assert search.status_code == 200
    results = search.json()
    assert len(results) == 1
    assert results[0]["document_id"] == body["document_id"]
    assert results[0]["chunk_index"] == 0
    assert results[0]["start_char"] == 0
    assert results[0]["end_char"] == len(payload.decode())
    assert results[0]["text"] == payload.decode()
    assert results[0]["score"] > 0


def test_search_can_be_scoped_to_uploaded_document(monkeypatch) -> None:
    client = fresh_client(monkeypatch)
    first = client.post(
        "/documents/upload",
        files={"file": ("first.txt", b"redis cache cache", "text/plain")},
    ).json()
    second = client.post(
        "/documents/upload",
        files={"file": ("second.txt", b"postgres database database", "text/plain")},
    ).json()

    response = client.post(
        "/search",
        json={"query": "redis cache", "document_id": second["document_id"]},
    )
    assert response.status_code == 200
    assert {hit["document_id"] for hit in response.json()} <= {second["document_id"]}
    assert first["document_id"] != second["document_id"]


def test_ready_reports_embedding_store_and_metadata_configuration(monkeypatch) -> None:
    client = fresh_client(monkeypatch)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "embedding_provider": "hashing",
        "embedding_model": "sha256-token-hashing-v1-64d",
        "embedding_dimensions": 64,
        "answer_provider": "extractive",
        "vector_store": "memory",
        "document_store": "memory",
        "embedding_cache": "memory",
    }


def test_ready_rejects_dimension_mismatch(monkeypatch) -> None:
    client = fresh_client(monkeypatch)
    monkeypatch.setattr(main, "vector_store", InMemoryVectorStore(dimensions=32))

    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding and vector-store dimensions differ."


def test_ready_rejects_unavailable_configured_cache(monkeypatch) -> None:
    client = fresh_client(monkeypatch)

    class UnavailableCache(InMemoryEmbeddingCache):
        def is_ready(self) -> bool:
            return False

    monkeypatch.setattr(main, "embedding_cache", UnavailableCache())

    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding cache is unavailable."


def test_get_document_returns_404_for_unknown_id(monkeypatch) -> None:
    client = fresh_client(monkeypatch)

    response = client.get("/documents/missing")

    assert response.status_code == 404


def test_delete_document_removes_metadata_vectors_and_cited_context(monkeypatch) -> None:
    client = fresh_client(monkeypatch)
    document = client.post(
        "/documents/upload",
        files={"file": ("delete.txt", b"PostgreSQL stores metadata.", "text/plain")},
    ).json()

    deleted = client.delete(f"/documents/{document['document_id']}")

    assert deleted.status_code == 204
    assert deleted.content == b""
    assert client.get(f"/documents/{document['document_id']}").status_code == 404
    search = client.post(
        "/search",
        json={"query": "metadata", "document_id": document["document_id"]},
    )
    assert search.status_code == 200
    assert search.json() == []
    question = client.post(
        "/questions",
        json={"question": "Where is metadata?", "document_id": document["document_id"]},
    )
    assert question.status_code == 200
    assert question.json()["status"] == "insufficient_context"
    assert question.json()["citations"] == []


def test_delete_unknown_document_returns_404(monkeypatch) -> None:
    client = fresh_client(monkeypatch)
    response = client.delete("/documents/missing")
    assert response.status_code == 404


def test_index_failure_compensates_persisted_metadata(monkeypatch) -> None:
    client = fresh_client(monkeypatch)

    class TrackingRepository(InMemoryDocumentRepository):
        last_saved_id: str | None = None

        def save(self, record):
            self.last_saved_id = record.document_id
            super().save(record)

    repository = TrackingRepository()

    class FailingVectorStore(InMemoryVectorStore):
        def replace_document(self, document_id, chunks, vectors):
            raise SQLAlchemyError("index unavailable")

    monkeypatch.setattr(main, "document_repository", repository)
    monkeypatch.setattr(main, "vector_store", FailingVectorStore(dimensions=64))

    response = client.post(
        "/documents/upload",
        files={"file": ("rollback.txt", b"durable source", "text/plain")},
    )

    assert response.status_code == 503
    assert repository.last_saved_id is not None
    assert repository.get(repository.last_saved_id) is None


def test_pdf_upload_enters_retrieval_and_cited_question_flow(monkeypatch) -> None:
    client = fresh_client(monkeypatch)
    upload = client.post(
        "/documents/upload",
        files={
            "file": (
                "report.pdf",
                build_text_pdf(
                    "PostgreSQL stores durable document metadata.",
                    "Citations retain page-aware source offsets.",
                ),
                "application/pdf",
            )
        },
    )
    assert upload.status_code == 201
    document_id = upload.json()["document_id"]

    search = client.post(
        "/search",
        json={"query": "Where is durable metadata stored?", "document_id": document_id},
    )
    assert search.status_code == 200
    assert {hit["page_start"] for hit in search.json()} == {1, 2}
    assert all(hit["page_start"] == hit["page_end"] for hit in search.json())

    question = client.post(
        "/questions",
        json={"question": "Where is durable metadata stored?", "document_id": document_id},
    )
    assert question.status_code == 200
    assert question.json()["status"] == "answered"
    assert question.json()["citations"][0]["page_start"] in {1, 2}


def test_search_and_question_reuse_model_scoped_query_embedding(monkeypatch) -> None:
    client = fresh_client(monkeypatch)

    class CountingEmbedder(HashingEmbedder):
        calls = 0

        def embed(self, texts):
            self.calls += 1
            return super().embed(texts)

    counting_embedder = CountingEmbedder(dimensions=64)
    monkeypatch.setattr(main, "embedder", counting_embedder)
    query = "Where is metadata stored?"
    document = client.post(
        "/documents/upload",
        files={"file": ("cache.txt", b"PostgreSQL stores metadata.", "text/plain")},
    ).json()

    client.post(
        "/search", json={"query": query, "document_id": document["document_id"]}
    ).raise_for_status()
    client.post(
        "/questions", json={"question": query, "document_id": document["document_id"]}
    ).raise_for_status()

    assert counting_embedder.calls == 2


def test_cache_failure_degrades_to_live_embedding(monkeypatch) -> None:
    client = fresh_client(monkeypatch)

    class FailingCache(InMemoryEmbeddingCache):
        def get(self, text, *, model_version, dimensions):
            raise RedisError("unavailable")

        def set(self, text, vector, *, model_version, dimensions):
            raise RedisError("unavailable")

    monkeypatch.setattr(main, "embedding_cache", FailingCache())
    document = client.post(
        "/documents/upload",
        files={"file": ("fallback.txt", b"PostgreSQL stores metadata.", "text/plain")},
    ).json()

    response = client.post(
        "/search",
        json={"query": "metadata", "document_id": document["document_id"]},
    )

    assert response.status_code == 200
    assert response.json()


def test_upload_surfaces_invalid_utf8_as_unprocessable_document(monkeypatch) -> None:
    client = fresh_client(monkeypatch)
    response = client.post(
        "/documents/upload",
        files={"file": ("broken.txt", b"prefix\xff", "text/plain")},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Document must be valid UTF-8 text."


def test_search_rejects_whitespace_only_query(monkeypatch) -> None:
    client = fresh_client(monkeypatch)
    response = client.post("/search", json={"query": "   "})
    assert response.status_code == 422
    assert response.json()["detail"] == "Search query must contain readable text."


def test_question_answer_returns_exact_traceable_citation(monkeypatch) -> None:
    client = fresh_client(monkeypatch)
    payload = b"PostgreSQL stores durable document metadata and citation spans."
    document = client.post(
        "/documents/upload",
        files={"file": ("architecture.txt", payload, "text/plain")},
    ).json()

    response = client.post(
        "/questions",
        json={
            "question": "Where is durable document metadata stored?",
            "document_id": document["document_id"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["answer_provider"] == "extractive"
    assert body["answer"] == payload.decode()
    assert body["citations"] == [
        {
            "document_id": document["document_id"],
            "chunk_index": 0,
            "text": payload.decode(),
            "start_char": 0,
            "end_char": len(payload.decode()),
            "page_start": None,
            "page_end": None,
            "score": body["citations"][0]["score"],
        }
    ]


def test_question_answer_abstains_without_indexed_context(monkeypatch) -> None:
    client = fresh_client(monkeypatch)

    response = client.post("/questions", json={"question": "What is the retention policy?"})

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_context"
    assert response.json()["citations"] == []


def test_requests_include_correlation_id_and_metrics_use_route_templates(
    monkeypatch, caplog
) -> None:
    client = fresh_client(monkeypatch)
    unknown_document_id = str(uuid4())

    response = client.get(f"/documents/{unknown_document_id}")
    metrics = client.get("/metrics")

    assert response.status_code == 404
    UUID(response.headers["X-Request-ID"])
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    assert unknown_document_id not in metrics.text
    completion_logs = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "document_intelligence.http"
    ]
    document_log = next(
        item for item in completion_logs if item["route"] == "/documents/{document_id}"
    )
    assert document_log["request_id"] == response.headers["X-Request-ID"]
    assert document_log["status"] == 404
    assert unknown_document_id not in json.dumps(completion_logs)
    families = {
        family.name: family for family in text_string_to_metric_families(metrics.text)
    }
    samples = families["document_intelligence_http_requests"].samples
    assert any(
        sample.labels
        == {
            "method": "GET",
            "route": "/documents/{document_id}",
            "status_class": "4xx",
        }
        and sample.value >= 1
        for sample in samples
        if sample.name == "document_intelligence_http_requests_total"
    )


def test_pipeline_metrics_report_formats_stages_retrieval_and_answer_status(monkeypatch) -> None:
    client = fresh_client(monkeypatch)
    document = client.post(
        "/documents/upload",
        files={"file": ("metrics.txt", b"PostgreSQL stores citation spans.", "text/plain")},
    ).json()
    client.post(
        "/search",
        json={"query": "citation spans", "document_id": document["document_id"]},
    ).raise_for_status()
    client.post(
        "/questions",
        json={
            "question": "Where are citation spans stored?",
            "document_id": document["document_id"],
        },
    ).raise_for_status()

    metrics = client.get("/metrics").text

    assert 'document_intelligence_ingestion_stages_total{format="txt",outcome="success",stage="completed"}' in metrics
    assert 'document_intelligence_retrieval_requests_total{operation="search",outcome="success"}' in metrics
    assert 'document_intelligence_retrieval_requests_total{operation="question",outcome="success"}' in metrics
    assert 'document_intelligence_answer_outcomes_total{provider="extractive",status="answered"}' in metrics
    assert 'document_intelligence_embedding_cache_operations_total{operation="get",outcome="miss"}' in metrics


def test_parse_failure_is_observable_without_document_identity(monkeypatch) -> None:
    client = fresh_client(monkeypatch)

    response = client.post(
        "/documents/upload",
        files={"file": ("private-name.txt", b"prefix\xff", "text/plain")},
    )
    metrics = client.get("/metrics").text

    assert response.status_code == 422
    assert 'document_intelligence_ingestion_stages_total{format="txt",outcome="error",stage="parse"}' in metrics
    assert "private-name.txt" not in metrics
