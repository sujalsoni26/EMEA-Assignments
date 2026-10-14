"""API tests with a fake Ollama client."""

from __future__ import annotations

import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.ollama_client import ModelNotFoundError, OllamaTimeoutError, OllamaUnavailableError
from tests.conftest import LOREM


@pytest.fixture
def client_with(fake_ollama):
    def _make(**kwargs):
        fake = fake_ollama(**kwargs)
        main.app.dependency_overrides[main.get_ollama] = lambda: fake
        return TestClient(main.app), fake

    yield _make
    main.app.dependency_overrides.clear()


def upload(client, data: bytes, filename="report.pdf", length="standard", stream=False):
    url = "/api/summarize/stream" if stream else "/api/summarize"
    return client.post(url, files={"file": (filename, data, "application/pdf")}, data={"length": length})


def test_index_and_config(client_with):
    with client_with()[0] as client:
        assert "PDF Summarization Agent" in client.get("/").text
        config = client.get("/api/config").json()
        assert config["model"] == "llama3"
        assert config["max_file_mb"] == 20


def test_health_reports_ollama_down(client_with):
    with client_with(error=OllamaUnavailableError("Cannot connect"))[0] as client:
        data = client.get("/api/health").json()
    assert data["status"] == "unavailable"


def test_summarize_returns_structured_summary(client_with, make_pdf):
    with client_with()[0] as client:
        response = upload(client, make_pdf([LOREM * 4, LOREM * 4]), length="brief")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Solar Power Overview"
    assert data["length"] == "brief"
    assert data["document"] == {"filename": "report.pdf", "pages": 2, "words": 8 * len(LOREM.split()), "title": None, "skipped_pages": []}
    assert data["stats"]["strategy"] == "single-pass"
    assert data["stats"]["model"] == "llama3"


def test_stream_emits_progress_then_result(client_with, make_pdf):
    with client_with()[0] as client:
        response = upload(client, make_pdf([LOREM * 4]), stream=True)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert [e["type"] for e in events[:-1]] == ["progress"] * (len(events) - 1)
    assert events[0]["stage"] == "extracted"
    assert events[-1]["type"] == "result"
    assert events[-1]["data"]["tldr"].startswith("Solar power")


def test_stream_reports_model_errors_as_events(client_with, make_pdf):
    with client_with(error=OllamaTimeoutError("Too slow"))[0] as client:
        response = upload(client, make_pdf([LOREM * 4]), stream=True)
    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert events[-1] == {"type": "error", "error": {"code": "model_timeout", "message": "Too slow"}}


@pytest.mark.parametrize(
    "error, status, code",
    [
        (OllamaUnavailableError("Cannot connect"), 503, "ollama_unavailable"),
        (ModelNotFoundError("Not installed"), 503, "model_not_found"),
        (OllamaTimeoutError("Too slow"), 504, "model_timeout"),
    ],
)
def test_model_errors(client_with, make_pdf, error, status, code):
    with client_with(error=error)[0] as client:
        response = upload(client, make_pdf([LOREM * 4]))
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_missing_file(client_with):
    with client_with()[0] as client:
        response = client.post("/api/summarize", data={"length": "brief"})
    assert response.status_code == 422
    assert response.json()["error"]["message"] == "No file was uploaded. Please choose a PDF document."


def test_invalid_length(client_with, make_pdf):
    with client_with()[0] as client:
        response = upload(client, make_pdf([LOREM]), length="huge")
    assert response.status_code == 422
    assert "brief" in response.json()["error"]["message"]


def test_non_pdf_upload(client_with):
    with client_with()[0] as client:
        response = upload(client, b"hello", filename="notes.txt")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "invalid_file_type"


def test_fake_pdf_upload(client_with):
    with client_with()[0] as client:
        response = upload(client, b"not a pdf at all", filename="fake.pdf")
    assert response.status_code == 415


def test_file_too_large(client_with, make_pdf, monkeypatch):
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, max_file_mb=0))
    with client_with()[0] as client:
        response = upload(client, make_pdf([LOREM]))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_scanned_pdf_is_rejected_before_calling_the_model(client_with):
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()  # a blank page has no text layer
    test_client, fake = client_with()
    with test_client as client:
        response = upload(client, doc.tobytes())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "no_extractable_text"
    assert fake.json_calls == [] and fake.chat_calls == []
