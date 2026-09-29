import pytest

from app.parsing import DocumentParseError, parse_text_document


def test_parse_text_document_normalizes_line_endings_and_metadata() -> None:
    parsed = parse_text_document(
        "notes.txt",
        b"first line\r\nsecond line  \rthird line\n",
    )

    assert parsed.filename == "notes.txt"
    assert parsed.extension == ".txt"
    assert parsed.text == "first line\nsecond line\nthird line"
    assert parsed.character_count == len(parsed.text)
    assert parsed.line_count == 3


def test_parse_markdown_document_preserves_markdown_content() -> None:
    payload = b"# Architecture\n\n- FastAPI\n- PostgreSQL\n"

    parsed = parse_text_document("architecture.md", payload)

    assert parsed.extension == ".md"
    assert parsed.text == "# Architecture\n\n- FastAPI\n- PostgreSQL"
    assert parsed.line_count == 4


@pytest.mark.parametrize("filename", ["report.pdf", "document.docx", "README"])
def test_parse_text_document_rejects_unsupported_extensions(filename: str) -> None:
    with pytest.raises(DocumentParseError, match="does not support extension"):
        parse_text_document(filename, b"content")


def test_parse_text_document_rejects_invalid_utf8() -> None:
    with pytest.raises(DocumentParseError, match="valid UTF-8"):
        parse_text_document("broken.txt", b"valid-prefix\xffinvalid")


@pytest.mark.parametrize("payload", [b"", b"   ", b"\r\n\t\r\n"])
def test_parse_text_document_rejects_empty_or_whitespace_only_content(
    payload: bytes,
) -> None:
    with pytest.raises(DocumentParseError, match="no readable text"):
        parse_text_document("empty.txt", payload)
