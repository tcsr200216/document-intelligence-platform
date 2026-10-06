"""Exercise the actual running HTTP service with a sample document.

Usage: python scripts/smoke_test.py [http://127.0.0.1:8000] [options]
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import httpx
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, StreamObject


def build_sample_pdf() -> bytes:
    writer = PdfWriter()
    for text in (
        "PostgreSQL stores durable document metadata and citation spans.",
        "Pgvector provides durable semantic retrieval across API restarts.",
    ):
        page = writer.add_blank_page(width=612, height=792)
        font_reference = writer._add_object(
            DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
        )
        stream = StreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def verify_document(client: httpx.Client, document_id: str, sample: bytes) -> tuple[int, int]:
    detail = client.get(f"/documents/{document_id}")
    detail.raise_for_status()
    assert detail.json()["sha256"] == hashlib.sha256(sample).hexdigest()
    inventory = client.get("/documents", params={"limit": 100, "status_filter": "indexed"})
    inventory.raise_for_status()
    assert document_id in {item["document_id"] for item in inventory.json()["items"]}
    query = f"Where are document metadata and citation spans stored? smoke-verification-{uuid4()}"

    response = client.post(
        "/search",
        json={
            "query": query,
            "document_id": document_id,
            "limit": 3,
        },
    )
    response.raise_for_status()
    hits = response.json()
    assert hits and all(hit["document_id"] == document_id for hit in hits)
    assert all(0 <= hit["start_char"] < hit["end_char"] and hit["text"] for hit in hits)
    assert all(hit["page_start"] == hit["page_end"] in {1, 2} for hit in hits)

    question = client.post(
        "/questions",
        json={
            "question": query,
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
        and citation["page_start"] == citation["page_end"] in {1, 2}
        for citation in answer["citations"]
    )
    return len(hits), len(answer["citations"])


def verify_deletion(client: httpx.Client, document_id: str, query: str) -> None:
    deleted = client.delete(f"/documents/{document_id}")
    deleted.raise_for_status()
    assert deleted.status_code == 204 and not deleted.content
    assert client.get(f"/documents/{document_id}").status_code == 404

    search = client.post(
        "/search",
        json={"query": query, "document_id": document_id, "limit": 3},
    )
    search.raise_for_status()
    assert search.json() == []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8000")
    parser.add_argument("--document-id-output", type=Path)
    parser.add_argument("--verify-document-id-file", type=Path)
    parser.add_argument("--delete-after-verify", action="store_true")
    args = parser.parse_args()
    sample = build_sample_pdf()
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=15.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json()["status"] == "ok"
        assert health.headers.get("X-Request-ID")

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
                files={"file": ("sample_document.pdf", sample, "application/pdf")},
            )
            upload.raise_for_status()
            document = upload.json()
            assert document["status"] == "indexed"
            assert document["chunk_count"] > 0
            document_id = document["document_id"]
            retry = client.post(
                "/documents/upload",
                files={"file": ("renamed-sample.pdf", sample, "application/pdf")},
            )
            retry.raise_for_status()
            assert retry.status_code == 200
            assert retry.json()["status"] == "already_indexed"
            assert retry.json()["document_id"] == document_id
            if args.document_id_output:
                args.document_id_output.write_text(document_id + "\n")

        hit_count, citation_count = verify_document(client, document_id, sample)

        if args.delete_after_verify:
            verify_deletion(client, document_id, "document metadata citation spans")

        metrics = client.get("/metrics")
        metrics.raise_for_status()
        assert "document_intelligence_http_requests_total" in metrics.text
        assert "document_intelligence_ingestion_stages_total" in metrics.text
        assert "document_intelligence_document_uploads_total" in metrics.text
        assert (
            'document_intelligence_embedding_cache_operations_total{operation="get",outcome="miss"}'
            in metrics.text
        )
        assert (
            'document_intelligence_embedding_cache_operations_total{operation="get",outcome="hit"}'
            in metrics.text
        )
        assert document_id not in metrics.text
        if args.delete_after_verify:
            assert "document_intelligence_document_deletions_total" in metrics.text

    print(
        f"Smoke test passed: document {document_id}, "
        f"inventory discovery, {hit_count} search hit(s), "
        f"{citation_count} answer citation(s)."
    )


if __name__ == "__main__":
    main()
