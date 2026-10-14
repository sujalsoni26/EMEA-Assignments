"""Split extracted PDF pages into retrieval chunks.

Chunks are built page by page (a chunk never spans two pages, so every answer
can cite an exact page). Within a page, whole paragraphs are packed together up
to ``chunk_chars`` characters, long paragraphs are split on sentence boundaries,
and consecutive chunks share a small overlap so facts on a boundary are not lost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .pdf_extractor import PageText

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")


@dataclass
class TextChunk:
    page: int
    text: str


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _pieces(paragraph: str, max_chars: int) -> list[str]:
    """Break a paragraph into pieces no longer than ``max_chars``."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces: list[str] = []
    for sentence in split_sentences(paragraph):
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        words, current = sentence.split(), ""
        for word in words:
            if current and len(current) + len(word) + 1 > max_chars:
                pieces.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            pieces.append(current)
    return pieces


def _tail(text: str, overlap_chars: int) -> str:
    """The last few sentences of ``text`` (at most ``overlap_chars`` characters)."""
    if overlap_chars <= 0:
        return ""
    tail: list[str] = []
    size = 0
    for sentence in reversed(split_sentences(text)):
        if size + len(sentence) > overlap_chars:
            break
        tail.insert(0, sentence)
        size += len(sentence) + 1
    return " ".join(tail)


def chunk_page(page: PageText, chunk_chars: int, overlap_chars: int) -> list[TextChunk]:
    pieces = [p for para in page.text.split("\n\n") if para.strip() for p in _pieces(para.strip(), chunk_chars - overlap_chars)]
    chunks: list[TextChunk] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n{piece}" if current else piece
        if current and len(candidate) > chunk_chars:
            chunks.append(TextChunk(page.number, current))
            overlap = _tail(current, overlap_chars)
            current = f"{overlap}\n{piece}" if overlap else piece
        else:
            current = candidate
    if current:
        chunks.append(TextChunk(page.number, current))
    return chunks


def chunk_document(pages: list[PageText], chunk_chars: int, overlap_chars: int) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for page in pages:
        chunks.extend(chunk_page(page, chunk_chars, overlap_chars))
    # Very short fragments (e.g. a lone heading at the bottom of a page) add noise to retrieval.
    return [c for c in chunks if len(c.text.split()) >= 5]
