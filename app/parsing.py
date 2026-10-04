from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class DocumentParseError(ValueError):
    """Raised when an uploaded document cannot be parsed safely."""


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    filename: str
    extension: str
    text: str
    character_count: int
    line_count: int
    pages: tuple[SourcePage, ...] = ()


@dataclass(frozen=True, slots=True)
class SourcePage:
    """One readable PDF page mapped into the normalized document text."""

    page_number: int
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A deterministic text chunk with stable offsets into normalized source text."""

    index: int
    text: str
    start_char: int
    end_char: int
    page_start: int | None = None
    page_end: int | None = None


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def parse_text_document(filename: str, payload: bytes) -> ParsedDocument:
    """Parse UTF-8 text or Markdown into normalized plain text metadata."""
    extension = Path(filename).suffix.lower()
    if extension not in {".txt", ".md"}:
        raise DocumentParseError(
            f"Text parser does not support extension '{extension or 'none'}'."
        )

    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentParseError("Document must be valid UTF-8 text.") from exc

    normalized = _normalize_text(decoded)

    if not normalized:
        raise DocumentParseError("Document contains no readable text.")

    return ParsedDocument(
        filename=filename,
        extension=extension,
        text=normalized,
        character_count=len(normalized),
        line_count=len(normalized.splitlines()),
    )


def parse_pdf_document(filename: str, payload: bytes) -> ParsedDocument:
    """Extract page-aware normalized text from a PDF without OCR."""
    if Path(filename).suffix.lower() != ".pdf":
        raise DocumentParseError("PDF parser requires a .pdf filename.")

    try:
        reader = PdfReader(BytesIO(payload), strict=False)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise DocumentParseError("Encrypted PDF documents are not supported.")
        extracted_pages = [page.extract_text() or "" for page in reader.pages]
    except DocumentParseError:
        raise
    except (PdfReadError, OSError, ValueError, TypeError, KeyError) as exc:
        raise DocumentParseError("Document is not a readable PDF.") from exc

    page_texts: list[str] = []
    page_numbers: list[int] = []
    for page_number, extracted in enumerate(extracted_pages, start=1):
        normalized = _normalize_text(extracted)
        if normalized:
            page_texts.append(normalized)
            page_numbers.append(page_number)

    if not page_texts:
        raise DocumentParseError(
            "PDF contains no extractable text; scanned PDFs require OCR, which is not enabled."
        )

    text = "\n\n".join(page_texts)
    pages: list[SourcePage] = []
    cursor = 0
    for page_number, page_text in zip(page_numbers, page_texts, strict=True):
        pages.append(SourcePage(page_number, cursor, cursor + len(page_text)))
        cursor += len(page_text) + 2

    return ParsedDocument(
        filename=filename,
        extension=".pdf",
        text=text,
        character_count=len(text),
        line_count=len(text.splitlines()),
        pages=tuple(pages),
    )


def parse_document(filename: str, payload: bytes) -> ParsedDocument:
    """Dispatch to the parser selected by the normalized file extension."""
    if Path(filename).suffix.lower() == ".pdf":
        return parse_pdf_document(filename, payload)
    return parse_text_document(filename, payload)


def chunk_text(
    text: str,
    *,
    max_chars: int = 1_000,
    overlap_chars: int = 150,
) -> list[TextChunk]:
    """Split normalized text into deterministic overlapping chunks.

    Character offsets are preserved so later retrieval results can be traced
    back to the exact span of normalized source text used for a citation.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")

    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative.")

    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars.")

    if not text:
        return []

    step = max_chars - overlap_chars
    chunks: list[TextChunk] = []

    for index, start_char in enumerate(range(0, len(text), step)):
        end_char = min(start_char + max_chars, len(text))
        chunks.append(
            TextChunk(
                index=index,
                text=text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
            )
        )

        if end_char == len(text):
            break

    return chunks


def chunk_document(
    document: ParsedDocument,
    *,
    max_chars: int = 1_000,
    overlap_chars: int = 150,
) -> list[TextChunk]:
    """Chunk a document without crossing PDF page boundaries."""
    if not document.pages:
        return chunk_text(
            document.text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

    chunks: list[TextChunk] = []
    for page in document.pages:
        page_text = document.text[page.start_char : page.end_char]
        local_chunks = chunk_text(
            page_text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        for local in local_chunks:
            chunks.append(
                TextChunk(
                    index=len(chunks),
                    text=local.text,
                    start_char=page.start_char + local.start_char,
                    end_char=page.start_char + local.end_char,
                    page_start=page.page_number,
                    page_end=page.page_number,
                )
            )
    return chunks
