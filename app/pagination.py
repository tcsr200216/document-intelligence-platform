from __future__ import annotations

import base64
import binascii
import json
from datetime import UTC, datetime

from app.repository import DocumentCursor


class InvalidDocumentCursor(ValueError):
    """Raised when a document-list cursor is malformed or used with another filter."""


def encode_document_cursor(cursor: DocumentCursor, status_filter: str | None) -> str:
    payload = json.dumps(
        {
            "document_id": cursor.document_id,
            "status": status_filter,
            "uploaded_at": cursor.uploaded_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "version": 1,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_document_cursor(token: str, status_filter: str | None) -> DocumentCursor:
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.b64decode(padded, altchars=b"-_", validate=True)
        payload = json.loads(raw)
        if not isinstance(payload, dict) or set(payload) != {
            "document_id",
            "status",
            "uploaded_at",
            "version",
        }:
            raise ValueError
        if payload["version"] != 1 or payload["status"] != status_filter:
            raise ValueError
        document_id = payload["document_id"]
        uploaded_at_value = payload["uploaded_at"]
        if (
            not isinstance(document_id, str)
            or not document_id
            or len(document_id) > 128
            or not isinstance(uploaded_at_value, str)
        ):
            raise ValueError
        uploaded_at = datetime.fromisoformat(uploaded_at_value)
        if uploaded_at.tzinfo is None or uploaded_at.utcoffset() is None:
            raise ValueError
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise InvalidDocumentCursor("Invalid or filter-mismatched document cursor.") from exc
    return DocumentCursor(uploaded_at=uploaded_at.astimezone(UTC), document_id=document_id)
