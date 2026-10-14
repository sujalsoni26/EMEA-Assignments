"""Tests for page-aware text segmentation."""

from __future__ import annotations

from app.chunker import CHARS_PER_TOKEN, chunk_pages, estimate_tokens
from app.pdf_extractor import PageText


def make_pages(n_pages: int, paragraphs_per_page: int = 4) -> list[PageText]:
    pages = []
    for p in range(1, n_pages + 1):
        paragraphs = [
            f"Page {p} paragraph {i}. " + "This sentence adds some content to the paragraph. " * 6
            for i in range(paragraphs_per_page)
        ]
        pages.append(PageText(number=p, text="\n\n".join(paragraphs)))
    return pages


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 400) == 100


def test_small_document_is_one_chunk():
    chunks = chunk_pages(make_pages(1, 2), chunk_tokens=3000, overlap_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].start_page == chunks[0].end_page == 1


def test_chunks_respect_budget_and_track_pages():
    pages = make_pages(12)
    chunks = chunk_pages(pages, chunk_tokens=400, overlap_tokens=40)
    assert len(chunks) > 3
    assert all(len(c.text) <= 400 * CHARS_PER_TOKEN + 40 * CHARS_PER_TOKEN for c in chunks)
    assert chunks[0].start_page == 1
    assert chunks[-1].end_page == 12
    assert all(a.start_page <= b.start_page for a, b in zip(chunks, chunks[1:]))
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_every_paragraph_appears_in_some_chunk():
    pages = make_pages(6)
    chunks = chunk_pages(pages, chunk_tokens=300, overlap_tokens=30)
    joined = "\n".join(c.text for c in chunks)
    for page in pages:
        for paragraph in page.text.split("\n\n"):
            assert paragraph.split(". ")[0] in joined  # each paragraph's first sentence is present


def test_consecutive_chunks_overlap():
    chunks = chunk_pages(make_pages(6), chunk_tokens=300, overlap_tokens=30)
    for previous, current in zip(chunks, chunks[1:]):
        last_sentence = previous.text.rstrip().split(". ")[-1]
        assert last_sentence in current.text


def test_very_long_paragraph_is_split():
    pages = [PageText(number=1, text="word " * 5000)]
    chunks = chunk_pages(pages, chunk_tokens=500, overlap_tokens=0)
    assert len(chunks) > 1
    assert sum(len(c.text.split()) for c in chunks) == 5000
