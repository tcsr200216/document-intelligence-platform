from datetime import UTC, datetime

import pytest

from app.pagination import (
    InvalidDocumentCursor,
    decode_document_cursor,
    encode_document_cursor,
)
from app.repository import DocumentCursor


def test_document_cursor_round_trips_position_and_filter() -> None:
    cursor = DocumentCursor(datetime(2026, 10, 5, 20, 0, tzinfo=UTC), "doc-123")

    token = encode_document_cursor(cursor, "indexed")

    assert decode_document_cursor(token, "indexed") == cursor
    assert "doc-123" not in token


def test_document_cursor_rejects_malformed_or_filter_mismatched_token() -> None:
    cursor = DocumentCursor(datetime(2026, 10, 5, 20, 0, tzinfo=UTC), "doc-123")
    token = encode_document_cursor(cursor, "indexed")

    with pytest.raises(InvalidDocumentCursor):
        decode_document_cursor("not-a-cursor", "indexed")
    with pytest.raises(InvalidDocumentCursor):
        decode_document_cursor(token, "indexing")
