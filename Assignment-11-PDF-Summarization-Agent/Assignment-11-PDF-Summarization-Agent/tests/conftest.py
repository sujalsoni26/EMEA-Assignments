"""Shared fixtures: PDF builders and a fake Ollama client (tests run offline)."""

from __future__ import annotations

import json

import pymupdf
import pytest

from app.ollama_client import ChatResult

LOREM = (
    "Solar panels convert sunlight into electricity using semiconductor cells. Prices fell sharply after "
    "governments introduced feed-in tariffs and manufacturers scaled up production. Grid operators now use "
    "batteries to store cheap midday power for the evening peak. "
)


def build_pdf(pages: list[str], *, header: str | None = None, footer: bool = False) -> bytes:
    doc = pymupdf.open()
    for i, text in enumerate(pages):
        page = doc.new_page()
        if header:
            page.insert_text((72, 40), header, fontsize=8)
        page.insert_textbox(page.rect + (72, 72, -72, -72), text, fontsize=10)
        if footer:
            page.insert_text((280, 810), f"Page {i + 1} of {len(pages)}", fontsize=8)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def make_pdf():
    return build_pdf


class FakeOllama:
    """Records calls and returns canned answers."""

    def __init__(self, *, error: Exception | None = None, final: dict | None = None):
        self.model = "llama3"
        self.error = error
        self.final = final or {
            "title": "Solar Power Overview",
            "tldr": "Solar power became cheap thanks to policy and scale.",
            "summary": "Solar power grew quickly.\n\nBatteries help the grid.",
            "key_points": ["Prices fell.", "Batteries store power.", "Prices fell."],
        }
        self.chat_calls: list[list[dict]] = []
        self.json_calls: list[list[dict]] = []

    async def is_model_available(self) -> bool:
        if self.error:
            raise self.error
        return True

    async def chat(self, messages, **kwargs) -> ChatResult:
        self.chat_calls.append(messages)
        if self.error:
            raise self.error
        return ChatResult(content="Here are the notes:\n- Point one about solar.\n- Point two about batteries.")

    async def chat_json(self, messages, schema, **kwargs) -> dict:
        self.json_calls.append(messages)
        if self.error:
            raise self.error
        return json.loads(json.dumps(self.final))


@pytest.fixture
def fake_ollama():
    return FakeOllama
