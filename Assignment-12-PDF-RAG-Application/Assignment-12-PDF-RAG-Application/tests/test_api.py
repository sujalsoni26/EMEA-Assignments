"""API tests with a fake Ollama client and a temporary knowledge base."""

from __future__ import annotations

import pytest

from app.ollama_client import OllamaUnavailableError
from app.rag import NO_ANSWER_MESSAGE
from tests.conftest import HANDBOOK_PAGES, FakeOllama


def upload(client, *files):
    return client.post("/api/documents", files=[("files", (name, data, "application/pdf")) for name, data in files])


def test_health_and_config(app_client):
    client, _ = app_client()
    with client:
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/config").json()["llm_model"] == "llama3"
        assert "PDF RAG Assistant" in client.get("/").text


def test_ask_without_documents(app_client):
    client, fake = app_client()
    with client:
        data = client.post("/api/ask", json={"question": "How much annual leave do I get?"}).json()
    assert data["status"] == "no_documents"
    assert fake.chat_messages == []


def test_upload_list_ask_and_delete(app_client, make_pdf):
    client, fake = app_client()
    with client:
        response = upload(client, ("handbook.pdf", make_pdf(HANDBOOK_PAGES)))
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["status"] == "indexed"
        assert result["document"]["pages"] == 2

        documents = client.get("/api/documents").json()
        assert len(documents["documents"]) == 1
        assert documents["total_chunks"] >= 2

        answer = client.post("/api/ask", json={"question": "How many days of paid annual leave do employees receive?", "top_k": 2}).json()
        assert answer["status"] == "answered"
        assert answer["answer"].endswith("[1].")
        assert answer["sources"][0]["page"] == 1
        assert answer["sources"][0]["cited"] is True
        assert answer["sources"][0]["filename"] == "handbook.pdf"

        # Only the retrieved passages and the question are sent to the LLM.
        prompt = fake.chat_messages[0][1]["content"]
        assert "25 days of paid annual leave" in prompt
        assert "Question: How many days of paid annual leave do employees receive?" in prompt

        doc_id = documents["documents"][0]["id"]
        assert client.delete(f"/api/documents/{doc_id}").status_code == 204
        assert client.get("/api/documents").json()["documents"] == []
        assert client.delete(f"/api/documents/{doc_id}").status_code == 404


def test_irrelevant_question_gets_safe_answer_without_llm_call(app_client, make_pdf):
    client, fake = app_client()
    with client:
        upload(client, ("handbook.pdf", make_pdf(HANDBOOK_PAGES)))
        data = client.post("/api/ask", json={"question": "Who painted the Mona Lisa?"}).json()
    assert data["status"] == "insufficient_context"
    assert data["answer"] == NO_ANSWER_MESSAGE
    assert all(not s["relevant"] for s in data["sources"])
    assert fake.chat_messages == []


def test_model_reporting_insufficient_context(app_client, make_pdf):
    client, _ = app_client(FakeOllama(answer="INSUFFICIENT_CONTEXT"))
    with client:
        upload(client, ("handbook.pdf", make_pdf(HANDBOOK_PAGES)))
        data = client.post("/api/ask", json={"question": "What is the annual leave for the CEO's dog?"}).json()
    assert data["status"] == "insufficient_context"
    assert data["answer"] == NO_ANSWER_MESSAGE
    assert data["sources"]  # the retrieved context is still returned


def test_duplicate_and_invalid_uploads(app_client, make_pdf):
    client, _ = app_client()
    pdf = make_pdf(HANDBOOK_PAGES)
    with client:
        upload(client, ("handbook.pdf", pdf))
        response = upload(client, ("copy.pdf", pdf), ("notes.txt", b"hello"), ("fake.pdf", b"not a pdf"))
        assert response.status_code == 207  # a duplicate is "already indexed", the others failed
        statuses = [r["status"] for r in response.json()["results"]]
        assert statuses == ["duplicate", "error", "error"]

        failed = upload(client, ("notes.txt", b"hello"))
        assert failed.status_code == 415  # every file failed: the first failure decides the status
        assert failed.json()["error"]["code"] == "invalid_file_type"

        mixed = upload(client, ("other.pdf", make_pdf(["A completely different document about office plants. " * 5])),
                       ("bad.pdf", b"nope"))
        assert mixed.status_code == 207
        assert [r["status"] for r in mixed.json()["results"]] == ["indexed", "error"]


def test_upload_without_files(app_client):
    client, _ = app_client()
    with client:
        response = client.post("/api/documents")
    assert response.status_code == 422
    assert "No file was uploaded" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"question": "  "}, "Please enter a question."),
        ({"question": "hi"}, "too short"),
        ({"question": "x" * 2000}, "too long"),
        ({"question": "Valid question?", "top_k": 50}, "top_k"),
        ({}, "'question' is required"),
    ],
)
def test_question_validation(app_client, payload, message):
    client, _ = app_client()
    with client:
        response = client.post("/api/ask", json=payload)
    assert response.status_code == 422
    assert message in response.json()["error"]["message"]


def test_ollama_down_during_upload_and_ask(app_client, make_pdf):
    client, fake = app_client()
    with client:
        upload(client, ("handbook.pdf", make_pdf(HANDBOOK_PAGES)))
        fake.error = OllamaUnavailableError("Cannot connect to Ollama")
        response = upload(client, ("other.pdf", make_pdf(["Different text about holidays and travel rules for staff. " * 4])))
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "ollama_unavailable"
        ask = client.post("/api/ask", json={"question": "How much annual leave?"})
        assert ask.status_code == 503
        assert client.get("/api/health").json()["status"] == "unavailable"
