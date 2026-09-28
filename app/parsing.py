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
