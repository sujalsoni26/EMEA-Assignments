"""Shared fixtures: PDF builder, a fake Ollama client with deterministic embeddings, and an app client."""

from __future__ import annotations

import dataclasses
import hashlib
import re

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings

HANDBOOK_PAGES = [
    "Annual leave. Full-time employees receive 25 days of paid annual leave per year. "
    "Unused leave days may be carried over until 31 March of the next year.",
    "Expenses. Hotel costs are reimbursed up to EUR 160 per night. Receipts are required for every "
    "expense above EUR 25. Expenses must be submitted within 30 days.",
]


def build_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(72, 72, 520, 770), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def bag_of_words_vector(text: str, dim: int = 64) -> list[float]:
    """Deterministic toy embedding: hashed bag of words, so texts sharing words are similar."""
    vector = np.zeros(dim)
    text = text.split(": ", 1)[-1]  # drop the nomic task prefix
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if len(word) > 3:
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    return vector.tolist() if vector.any() else [1.0] + [0.0] * (dim - 1)


class FakeOllama:
    def __init__(self, answer: str = "Full-time employees get 25 days of paid annual leave [1].", error: Exception | None = None):
        self.llm_model = "llama3"
        self.embedding_model = "nomic-embed-text"
        self.answer = answer
        self.error = error
        self.chat_messages: list[list[dict]] = []

    async def missing_models(self):
        if self.error:
            raise self.error
        return []

    async def embed(self, texts):
        if self.error:
            raise self.error
        return [bag_of_words_vector(t) for t in texts]

    async def chat(self, messages, **kwargs):
        self.chat_messages.append(messages)
        return self.answer

    async def aclose(self):
        return None


@pytest.fixture
def make_pdf():
    return build_pdf


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """Return a factory creating a TestClient whose app uses a fake Ollama and a temporary data dir."""
    test_settings = dataclasses.replace(settings, data_dir=tmp_path, min_similarity=0.3)
    monkeypatch.setattr(main, "settings", test_settings)

    def _make(fake: FakeOllama | None = None):
        fake = fake or FakeOllama()
        monkeypatch.setattr(main, "OllamaClient", lambda *args, **kwargs: fake)
        return TestClient(main.app), fake

    return _make
