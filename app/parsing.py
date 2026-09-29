from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class DocumentParseError(ValueError):
    """Raised when an uploaded document cannot be parsed safely."""


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    filename: str
    extension: str
    text: str
    character_count: int
    line_count: int


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A deterministic text chunk with stable offsets into normalized source text."""

    index: int
    text: str
    start_char: int
    end_char: int


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

    normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n")).strip()

    if not normalized:
        raise DocumentParseError("Document contains no readable text.")

    return ParsedDocument(
        filename=filename,
        extension=extension,
        text=normalized,
        character_count=len(normalized),
        line_count=len(normalized.splitlines()),
    )


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
