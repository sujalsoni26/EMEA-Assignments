"""API tests. Ollama is replaced by a fake client, so these tests run offline."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_ollama
from app.ollama_client import (
    ChatResult,
    EmptyResponseError,
    ModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)

SAMPLE_TEXT = (
    "Coral reefs cover less than one percent of the ocean floor, yet they support roughly a quarter "
    "of all known marine species. Reefs are built by tiny animals called coral polyps, which live in a "
    "close partnership with microscopic algae. When ocean water becomes too warm, corals expel the algae "
    "and turn white, a process known as bleaching."
)


class FakeOllama:
    def __init__(self, reply: str = "", error: Exception | None = None, available: bool = True):
        self.model = "tinyllama"
        self.reply = reply
        self.error = error
        self.available = available
        self.calls: list[list[dict]] = []

    async def chat(self, messages, options=None):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return ChatResult(content=self.reply)

    async def is_model_available(self):
        if self.error:
            raise self.error
        return self.available


@pytest.fixture
def make_client():
    clients = []

    def _make(fake: FakeOllama, raise_server_exceptions: bool = True) -> TestClient:
        app.dependency_overrides[get_ollama] = lambda: fake
        client = TestClient(app, raise_server_exceptions=raise_server_exceptions)
        clients.append(client)
        return client

    yield _make
    app.dependency_overrides.clear()


def test_index_page_is_served(make_client):
    client = make_client(FakeOllama())
    response = client.get("/")
    assert response.status_code == 200
    assert "Local LLM Text Summarizer" in response.text


def test_config_exposes_limits(make_client):
    data = make_client(FakeOllama()).get("/api/config").json()
    assert data["min_words"] == 30
    assert data["max_chars"] == 12000


def test_health_ok(make_client):
    data = make_client(FakeOllama()).get("/api/health").json()
    assert data["status"] == "ok"
    assert data["model_available"] is True


def test_health_reports_missing_model(make_client):
    data = make_client(FakeOllama(available=False)).get("/api/health").json()
    assert data["status"] == "degraded"
    assert "ollama pull tinyllama" in data["message"]


def test_health_reports_ollama_down(make_client):
    fake = FakeOllama(error=OllamaUnavailableError("Cannot connect to Ollama"))
    data = make_client(fake).get("/api/health").json()
    assert data["status"] == "unavailable"
    assert data["ollama_reachable"] is False


def test_summarize_paragraph_success(make_client):
    fake = FakeOllama(
        reply="Sure! Here is a summary of the text:\n\nCoral reefs support a quarter of marine species. "
        "Warm water causes bleaching."
    )
    response = make_client(fake).post("/api/summarize", json={"text": SAMPLE_TEXT})
    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == "Coral reefs support a quarter of marine species. Warm water causes bleaching."
    assert data["format"] == "paragraph"
    assert data["length"] == "medium"
    assert data["model"] == "tinyllama"
    assert data["chunks"] == 1
    assert data["input_words"] == len(SAMPLE_TEXT.split())
    assert data["summary_words"] == 12
    # The text must be sent to the model, followed by the instruction.
    user_message = fake.calls[0][-1]["content"]
    assert user_message.startswith(SAMPLE_TEXT)


def test_summarize_bullets_success(make_client):
    fake = FakeOllama(reply="1. Reefs host many species.\n2. Corals live with algae.\n3. Heat causes bleaching.")
    response = make_client(fake).post(
        "/api/summarize", json={"text": SAMPLE_TEXT, "format": "bullets", "length": "short"}
    )
    assert response.status_code == 200
    assert response.json()["summary"] == (
        "- Reefs host many species.\n- Corals live with algae.\n- Heat causes bleaching."
    )


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"text": "   "}, "Please enter some text"),
        ({"text": "Too short to summarize."}, "too short"),
        ({"text": "word " * 3000}, "too long"),
        ({"text": "123 " * 40}, "readable words"),
        ({}, "'text' is required"),
        ({"text": SAMPLE_TEXT, "length": "huge"}, "'length'"),
        ({"text": SAMPLE_TEXT, "format": "table"}, "'format'"),
    ],
)
def test_validation_errors(make_client, payload, expected):
    fake = FakeOllama(reply="unused")
    response = make_client(fake).post("/api/summarize", json=payload)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert expected in error["message"]
    assert fake.calls == []  # the model is never called for invalid input


def test_malformed_json(make_client):
    response = make_client(FakeOllama()).post(
        "/api/summarize", content='{"text": ', headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["message"] == "The request body is not valid JSON."


@pytest.mark.parametrize(
    "error, status, code",
    [
        (OllamaUnavailableError("Cannot connect to Ollama"), 503, "ollama_unavailable"),
        (ModelNotFoundError("Model missing"), 503, "model_not_found"),
        (OllamaTimeoutError("Too slow"), 504, "ollama_timeout"),
        (EmptyResponseError("Empty"), 502, "empty_response"),
    ],
)
def test_ollama_errors_are_reported_clearly(make_client, error, status, code):
    response = make_client(FakeOllama(error=error)).post("/api/summarize", json={"text": SAMPLE_TEXT})
    assert response.status_code == status
    assert response.json()["error"] == {"code": code, "message": error.message}


def test_unusable_model_output_returns_502(make_client):
    response = make_client(FakeOllama(reply="Sure!")).post("/api/summarize", json={"text": SAMPLE_TEXT})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "empty_response"


def test_unexpected_errors_return_generic_500(make_client):
    client = make_client(FakeOllama(error=RuntimeError("boom")), raise_server_exceptions=False)
    response = client.post("/api/summarize", json={"text": SAMPLE_TEXT})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "boom" not in response.text  # internal details are not leaked


def test_long_text_is_summarized_in_chunks(make_client):
    paragraph = SAMPLE_TEXT + "\n\n"
    long_text = paragraph * 12  # about 4,000 characters, above the 3,500 chunk size
    fake = FakeOllama(reply="Coral reefs are important and threatened by warming oceans.")
    response = make_client(fake).post("/api/summarize", json={"text": long_text})
    assert response.status_code == 200
    assert response.json()["chunks"] == 2
    assert len(fake.calls) == 3  # two chunk summaries + one final summary
