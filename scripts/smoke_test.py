"""Exercise the actual running HTTP service with a sample document.

Usage: python scripts/smoke_test.py [http://127.0.0.1:8000] [options]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import httpx


def verify_document(client: httpx.Client, document_id: str, sample: Path) -> tuple[int, int]:
    detail = client.get(f"/documents/{document_id}")
    detail.raise_for_status()
    assert detail.json()["sha256"] == hashlib.sha256(sample.read_bytes()).hexdigest()

    response = client.post(
        "/search",
        json={
            "query": "Where are document metadata and citation spans stored?",
            "document_id": document_id,
            "limit": 3,
        },
    )
    response.raise_for_status()
    hits = response.json()
    assert hits and all(hit["document_id"] == document_id for hit in hits)
    assert all(0 <= hit["start_char"] < hit["end_char"] and hit["text"] for hit in hits)

    question = client.post(
        "/questions",
        json={
            "question": "Where are document metadata and citation spans stored?",
            "document_id": document_id,
            "context_limit": 3,
        },
    )
    question.raise_for_status()
    answer = question.json()
    assert answer["status"] == "answered"
    assert answer["citations"]
    assert all(
        citation["document_id"] == document_id
        and 0 <= citation["start_char"] < citation["end_char"]
        and citation["text"]
        for citation in answer["citations"]
    )
    return len(hits), len(answer["citations"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8000")
    parser.add_argument("--document-id-output", type=Path)
    parser.add_argument("--verify-document-id-file", type=Path)
    args = parser.parse_args()
    sample = Path(__file__).resolve().parents[1] / "data" / "sample_document.txt"
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=15.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json()["status"] == "ok"

        ready = client.get("/ready")
        ready.raise_for_status()
        assert ready.json()["status"] == "ready"

        if args.verify_document_id_file:
            document_id = args.verify_document_id_file.read_text().strip()
            if not document_id:
                parser.error("document ID file is empty")
        else:
            upload = client.post(
                "/documents/upload",
                files={"file": ("sample_document.txt", sample.read_bytes(), "text/plain")},
            )
            upload.raise_for_status()
            document = upload.json()
            assert document["status"] == "indexed"
            assert document["chunk_count"] > 0
            document_id = document["document_id"]
            if args.document_id_output:
                args.document_id_output.write_text(document_id + "\n")

        hit_count, citation_count = verify_document(client, document_id, sample)

    print(
        f"Smoke test passed: document {document_id}, "
        f"{hit_count} search hit(s), {citation_count} answer citation(s)."
    )


if __name__ == "__main__":
    main()
