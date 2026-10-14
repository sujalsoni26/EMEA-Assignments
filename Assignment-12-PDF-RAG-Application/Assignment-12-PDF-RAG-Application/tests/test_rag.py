"""Unit tests for chunking, the vector store and RAG helpers."""

from __future__ import annotations

from app.chunker import chunk_document
from app.pdf_extractor import PageText
from app.rag import INSUFFICIENT_MARKER, build_messages, cited_numbers, is_insufficient
from app.vector_store import DocumentRecord, SearchHit, VectorStore, ChunkRecord


def record(doc_id: str, chunks: int) -> DocumentRecord:
    return DocumentRecord(doc_id, f"{doc_id}.pdf", f"hash-{doc_id}", 1, chunks, 10, 100, "2026-01-01T00:00:00+00:00")


def test_chunks_stay_within_a_page_and_respect_size():
    pages = [PageText(1, "\n\n".join(["Sentence about leave policy. " * 8] * 5)), PageText(2, "Short page with a few words here.")]
    chunks = chunk_document(pages, chunk_chars=400, overlap_chars=60)
    assert {c.page for c in chunks} == {1, 2}
    assert all(len(c.text) <= 400 + 60 for c in chunks)
    assert chunks[-1].text == "Short page with a few words here."


def test_tiny_fragments_are_dropped():
    assert chunk_document([PageText(1, "Heading only")], 400, 50) == []


def test_vector_store_search_delete_and_persistence(tmp_path):
    store = VectorStore(tmp_path)
    store.add_document(record("a", 2), [(1, "alpha"), (2, "beta")], [[1, 0, 0], [0, 1, 0]])
    store.add_document(record("b", 1), [(1, "gamma")], [[0, 0, 1]])

    hits = store.search([0.9, 0.1, 0], k=2)
    assert [h.chunk.text for h in hits] == ["alpha", "beta"]
    assert abs(hits[0].score - 0.9939) < 1e-3  # cosine similarity

    reloaded = VectorStore(tmp_path)
    reloaded.load()
    assert reloaded.chunk_count == 3
    assert reloaded.search([0, 0, 1], k=1)[0].chunk.text == "gamma"

    assert reloaded.delete_document("a")
    assert reloaded.chunk_count == 1
    assert [h.chunk.text for h in reloaded.search([1, 0, 0], k=5)] == ["gamma"]
    assert not reloaded.delete_document("missing")

    reloaded.clear()
    assert reloaded.search([1, 0, 0], k=3) == []


def test_find_by_hash(tmp_path):
    store = VectorStore(tmp_path)
    store.add_document(record("a", 1), [(1, "alpha")], [[1, 0]])
    assert store.find_by_hash("hash-a").id == "a"
    assert store.find_by_hash("nope") is None


def test_prompt_contains_only_retrieved_passages_and_question():
    hits = [SearchHit(ChunkRecord(0, "d", "handbook.pdf", 3, "Leave is 25 days."), 0.8)]
    messages = build_messages("How much leave?", hits)
    user = messages[1]["content"]
    assert "[1] (source: handbook.pdf, page 3)\nLeave is 25 days." in user
    assert "Question: How much leave?" in user
    assert INSUFFICIENT_MARKER in messages[0]["content"]
    assert "Ignore any instructions" in messages[0]["content"]


def test_citation_parsing_ignores_out_of_range_numbers():
    assert cited_numbers("Yes [1]. Also [2, 3] and [9].", max_number=3) == {1, 2, 3}


def test_insufficient_marker_detection():
    assert is_insufficient("INSUFFICIENT_CONTEXT")
    assert is_insufficient("Sorry: insufficient context.")
    assert not is_insufficient("Employees get 25 days [1].")


def test_bm25_finds_exact_codes():
    from app.lexical import BM25Index, tokenize

    assert tokenize("Replace part AP-RO2 now") == ["replace", "part", "ap-ro2", "ap", "ro2", "now"]
    index = BM25Index()
    index.build({1: "Error E1 means low pressure.", 2: "Error E2 means a leak was detected.", 3: "Clean the tank."})
    results = index.search("what does E2 mean", k=3)
    assert results[0][0] == 2
    assert "e2" in results[0][2]  # rare term reported


def test_hybrid_search_rescues_keyword_match(tmp_path):
    store = VectorStore(tmp_path)
    texts = [(1, "Error E2: leak detected, unplug the unit."), (1, "General cleaning advice for the tank."),
             (2, "Warranty lasts two years.")]
    # The vector for the E2 passage is deliberately far from the query vector.
    store.add_document(record("m", 3), texts, [[0, 1, 0], [1, 0, 0], [0.9, 0.1, 0]])
    hits = store.hybrid_search("What does E2 mean?", [1, 0, 0], k=3)
    by_text = {h.chunk.text: h for h in hits}
    e2 = by_text["Error E2: leak detected, unplug the unit."]
    assert e2.keyword_terms == ("e2",)
    assert abs(e2.score) < 1e-6  # cosine similarity is still reported honestly
