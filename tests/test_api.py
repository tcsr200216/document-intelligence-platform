from fastapi.testclient import TestClient

from app import main
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
        "vector_store": "memory",
        "document_store": "memory",
    }


def test_ready_rejects_dimension_mismatch(monkeypatch) -> None:
    client = fresh_client(monkeypatch)
    monkeypatch.setattr(main, "vector_store", InMemoryVectorStore(dimensions=32))

    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding and vector-store dimensions differ."


def test_get_document_returns_404_for_unknown_id(monkeypatch) -> None:
    client = fresh_client(monkeypatch)

    response = client.get("/documents/missing")

    assert response.status_code == 404


def test_upload_rejects_pdf_until_extraction_is_implemented(monkeypatch) -> None:
    client = fresh_client(monkeypatch)
    response = client.post(
        "/documents/upload",
        files={"file": ("report.pdf", b"%PDF-test", "application/pdf")},
    )
    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"]


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
