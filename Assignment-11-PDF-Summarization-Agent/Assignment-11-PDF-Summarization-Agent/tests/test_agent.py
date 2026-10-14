"""Tests for the summarization agent's planning and output handling."""

from __future__ import annotations

import dataclasses

import pytest

from app.agent import PDFSummarizationAgent, SummaryLength, normalise_final, parse_bullets
from app.config import settings
from app.ollama_client import InvalidModelOutputError
from app.pdf_extractor import ExtractedDocument, PageText
from tests.conftest import LOREM


def make_document(pages: int, repeat: int = 2) -> ExtractedDocument:
    return ExtractedDocument(
        filename="doc.pdf",
        page_count=pages,
        pages=[PageText(number=i + 1, text=(LOREM * repeat).strip()) for i in range(pages)],
    )


async def collect(agent, document, length=SummaryLength.standard):
    events = []

    async def record(event):
        events.append(event)

    result = await agent.run(document, length, record)
    return result, events


async def test_short_document_uses_single_pass(fake_ollama):
    client = fake_ollama()
    result, events = await collect(PDFSummarizationAgent(client, settings), make_document(2))
    assert result.strategy == "single-pass"
    assert result.chunks == 1
    assert client.chat_calls == []  # no map step
    assert len(client.json_calls) == 1
    assert "full text of the document" in client.json_calls[0][-1]["content"]
    assert [e.stage for e in events] == ["analyzing", "planning", "writing", "done"]
    assert result.key_points == ["Prices fell.", "Batteries store power."]  # duplicates removed
    assert result.summary == "Solar power grew quickly.\n\nBatteries help the grid."


async def test_long_document_uses_map_reduce(fake_ollama):
    client = fake_ollama()
    small = dataclasses.replace(settings, single_pass_tokens=200, chunk_tokens=300, chunk_overlap_tokens=20)
    result, events = await collect(PDFSummarizationAgent(client, small), make_document(6))
    assert result.strategy == "map-reduce"
    assert result.chunks > 1
    assert len(client.chat_calls) == result.chunks  # one map call per chunk
    assert len(result.sections) == result.chunks
    assert result.sections[0].notes == ["Point one about solar.", "Point two about batteries."]
    assert "notes covering the whole document" in client.json_calls[0][-1]["content"]
    mapping = [e for e in events if e.stage == "mapping"]
    assert [(e.current, e.total) for e in mapping] == [(i, result.chunks) for i in range(1, result.chunks + 1)]


async def test_notes_are_merged_when_they_exceed_the_budget(fake_ollama):
    client = fake_ollama()
    tiny = dataclasses.replace(
        settings, single_pass_tokens=100, chunk_tokens=150, chunk_overlap_tokens=10, notes_budget_tokens=40
    )
    result, events = await collect(PDFSummarizationAgent(client, tiny), make_document(8))
    assert any(e.stage == "reducing" for e in events)
    assert len(client.chat_calls) > result.chunks  # map calls + reduce calls
    assert result.llm_calls == len(client.chat_calls) + 1


def test_parse_bullets_handles_numbering_and_preamble():
    text = "Here are the notes:\n1. First point\n2) Second point\n* Third point\n• Fourth"
    assert parse_bullets(text) == ["First point", "Second point", "Third point", "Fourth"]


def test_normalise_final_accepts_key_points_as_string():
    data = {"title": "", "tldr": "", "summary": "Para one. More.\n\nPara two.", "key_points": "- A\n- B"}
    title, tldr, summary, points = normalise_final(data, SummaryLength.brief, "fallback")
    assert title == "fallback"
    assert tldr == "Para one."  # derived from the summary when missing
    assert points == ["A", "B"]


def test_normalise_final_rejects_incomplete_output():
    with pytest.raises(InvalidModelOutputError):
        normalise_final({"title": "x", "summary": "", "key_points": []}, SummaryLength.brief, "f")


def test_normalise_final_splits_paragraphs_on_single_newlines():
    data = {"title": "T", "tldr": "x", "summary": "First paragraph.\nSecond paragraph.", "key_points": ["A"]}
    assert normalise_final(data, SummaryLength.standard, "f")[2] == "First paragraph.\n\nSecond paragraph."
