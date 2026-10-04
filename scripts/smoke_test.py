"""Exercise the actual running HTTP service with a sample document.

Usage: python scripts/smoke_test.py [http://127.0.0.1:8000]
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx


def main(base_url: str = "http://127.0.0.1:8000") -> None:
    sample = Path(__file__).resolve().parents[1] / "data" / "sample_document.txt"
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=15.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json()["status"] == "ok"

        ready = client.get("/ready")
        ready.raise_for_status()
        assert ready.json()["status"] == "ready"

        upload = client.post(
            "/documents/upload",
            files={"file": ("sample_document.txt", sample.read_bytes(), "text/plain")},
        )
        upload.raise_for_status()
        document = upload.json()
        assert document["status"] == "indexed"
        assert document["chunk_count"] > 0

        detail = client.get(f"/documents/{document['document_id']}")
        detail.raise_for_status()
        assert detail.json()["sha256"] == document["sha256"]

        response = client.post(
            "/search",
            json={
                "query": "Where are document metadata and citation spans stored?",
                "document_id": document["document_id"],
                "limit": 3,
            },
        )
        response.raise_for_status()
        hits = response.json()
        assert hits and all(hit["document_id"] == document["document_id"] for hit in hits)
        assert all(
            0 <= hit["start_char"] < hit["end_char"] and hit["text"]
            for hit in hits
        )

        question = client.post(
            "/questions",
            json={
                "question": "Where are document metadata and citation spans stored?",
                "document_id": document["document_id"],
                "context_limit": 3,
            },
        )
        question.raise_for_status()
        answer = question.json()
        assert answer["status"] == "answered"
        assert answer["citations"]
        assert all(
            citation["document_id"] == document["document_id"]
            and 0 <= citation["start_char"] < citation["end_char"]
            and citation["text"]
            for citation in answer["citations"]
        )

    print(
        f"Smoke test passed: document {document['document_id']}, "
        f"{len(hits)} search hit(s), {len(answer['citations'])} answer citation(s)."
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000")
