import pytest

from app.parsing import DocumentParseError, chunk_text, parse_text_document


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


def test_chunk_text_creates_overlapping_chunks_with_source_offsets() -> None:
    text = "abcdefghij"

    chunks = chunk_text(text, max_chars=4, overlap_chars=1)

    assert [(chunk.index, chunk.text, chunk.start_char, chunk.end_char) for chunk in chunks] == [
        (0, "abcd", 0, 4),
        (1, "defg", 3, 7),
        (2, "ghij", 6, 10),
    ]
    assert all(text[chunk.start_char:chunk.end_char] == chunk.text for chunk in chunks)


def test_chunk_text_returns_single_chunk_when_text_fits() -> None:
    chunks = chunk_text("short text", max_chars=100, overlap_chars=10)

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len("short text")
    assert chunks[0].text == "short text"


def test_chunk_text_returns_empty_list_for_empty_text() -> None:
    assert chunk_text("", max_chars=10, overlap_chars=2) == []


@pytest.mark.parametrize(
    ("max_chars", "overlap_chars", "message"),
    [
        (0, 0, "max_chars must be greater than zero"),
        (-1, 0, "max_chars must be greater than zero"),
        (10, -1, "overlap_chars cannot be negative"),
        (10, 10, "overlap_chars must be smaller than max_chars"),
        (10, 11, "overlap_chars must be smaller than max_chars"),
    ],
)
def test_chunk_text_rejects_invalid_configuration(
    max_chars: int,
    overlap_chars: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        chunk_text("content", max_chars=max_chars, overlap_chars=overlap_chars)
