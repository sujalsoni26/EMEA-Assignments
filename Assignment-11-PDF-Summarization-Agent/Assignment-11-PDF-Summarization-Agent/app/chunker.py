"""Text segmentation for long documents.

Chunks are built from whole paragraphs where possible (falling back to sentences
and then words for very long paragraphs), are limited to a token budget, overlap
slightly with the previous chunk, and remember which pages they came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .pdf_extractor import PageText

# Rough rule of thumb for English text with the Llama 3 tokenizer.
CHARS_PER_TOKEN = 4

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN)) if text else 0


@dataclass
class Chunk:
    index: int
    text: str
    start_page: int
    end_page: int

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)

    @property
    def page_label(self) -> str:
        if self.start_page == self.end_page:
            return f"page {self.start_page}"
        return f"pages {self.start_page}–{self.end_page}"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split a paragraph that is longer than ``max_chars`` into smaller pieces."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in split_sentences(text):
        if len(sentence) > max_chars:  # extremely long "sentence" (tables, lists...)
            words = sentence.split()
            sentence_parts, part = [], ""
            for word in words:
                if part and len(part) + len(word) + 1 > max_chars:
                    sentence_parts.append(part)
                    part = word
                else:
                    part = f"{part} {word}" if part else word
            if part:
                sentence_parts.append(part)
        else:
            sentence_parts = [sentence]
        for part in sentence_parts:
            if current and len(current) + len(part) + 1 > max_chars:
                pieces.append(current)
                current = part
            else:
                current = f"{current} {part}" if current else part
    if current:
        pieces.append(current)
    return pieces


def _overlap_tail(units: list[tuple[int, str]], overlap_chars: int) -> list[tuple[int, str]]:
    """Return the last sentences of ``units`` (up to ``overlap_chars``) to start the next chunk."""
    if overlap_chars <= 0:
        return []
    tail: list[tuple[int, str]] = []
    size = 0
    for page, text in reversed(units):
        for sentence in reversed(split_sentences(text) or [text]):
            if size + len(sentence) > overlap_chars:
                return tail
            tail.insert(0, (page, sentence))
            size += len(sentence) + 1
    return tail


def chunk_pages(pages: list[PageText], chunk_tokens: int, overlap_tokens: int) -> list[Chunk]:
    max_chars = chunk_tokens * CHARS_PER_TOKEN
    overlap_chars = min(overlap_tokens * CHARS_PER_TOKEN, max_chars // 4)

    units: list[tuple[int, str]] = []
    for page in pages:
        for paragraph in page.text.split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph:
                units.extend((page.number, piece) for piece in _split_long(paragraph, max_chars - overlap_chars))

    chunks: list[Chunk] = []
    current: list[tuple[int, str]] = []
    size = 0
    has_new_content = False

    def flush() -> None:
        chunks.append(
            Chunk(
                index=len(chunks),
                text="\n\n".join(text for _, text in current),
                start_page=current[0][0],
                end_page=current[-1][0],
            )
        )

    for unit in units:
        unit_size = len(unit[1]) + 2
        if current and has_new_content and size + unit_size > max_chars:
            flush()
            current = _overlap_tail(current, overlap_chars)
            size = sum(len(text) + 2 for _, text in current)
            has_new_content = False
        current.append(unit)
        size += unit_size
        has_new_content = True

    if current and has_new_content:
        flush()
    return chunks
