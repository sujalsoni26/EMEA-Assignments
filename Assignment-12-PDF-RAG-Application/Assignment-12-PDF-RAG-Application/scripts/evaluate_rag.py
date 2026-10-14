"""Evaluate the RAG pipeline on the sample documents.

    python scripts/evaluate_rag.py               # full evaluation (needs Ollama running)
    python scripts/evaluate_rag.py --retrieval-only

The script indexes every PDF in ``samples/`` into a temporary knowledge base and
asks a fixed set of questions:

* **answerable** questions, each with the document and page that contains the
  answer and keywords the answer must mention;
* **unanswerable** questions, whose answers are not in the documents, where the
  correct behaviour is a safe "not enough information" reply.

It reports retrieval hit rate (is the right page among the retrieved passages?),
answer accuracy, refusal accuracy and the similarity-score distribution used to
choose ``MIN_SIMILARITY``. Results are written to ``reports/rag_evaluation.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.ollama_client import OllamaClient  # noqa: E402
from app.rag import QUERY_PREFIX, RAGService  # noqa: E402
from app.vector_store import VectorStore  # noqa: E402

HANDBOOK = "northwind_employee_handbook.pdf"
MANUAL = "aquapure_x2_user_manual.pdf"
NIST = "NIST_AI_RMF_1.0.pdf"

# (question, document containing the answer, evidence phrase found on the answer page,
#  keywords that must appear in the generated answer)
ANSWERABLE = [
    ("How many days of paid annual leave do full-time employees get?", HANDBOOK, "25 days of paid annual leave", ["25"]),
    ("What are the core working hours at Northwind Labs?", HANDBOOK, "core hours from 10:00 to 15:00", ["10:00", "15:00"]),
    ("For how many days per year can I work remotely from another country?", HANDBOOK, "up to 20 working days", ["20"]),
    ("What is the maximum hotel cost per night in London?", HANDBOOK, "London or Paris", ["220"]),
    ("How much is the yearly learning budget?", HANDBOOK, "EUR 1,500", ["1,500"]),
    ("How long is the partner's paid parental leave?", HANDBOOK, "receives 6 weeks of fully paid leave", ["6 weeks"]),
    ("What is the notice period after two years of employment?", HANDBOOK, "two months after that", ["two months"]),
    ("How often should the RO membrane of the AquaPure X2 be replaced?", MANUAL, "AP-RO2", ["24 months"]),
    ("What does error code E2 mean on the purifier?", MANUAL, "Leak detected. Unplug", ["leak"]),
    ("What does E2 mean and what should I do?", MANUAL, "Leak detected. Unplug", ["leak"]),
    ("What is the part number of the RO membrane?", MANUAL, "AP-RO2", ["AP-RO2"]),
    ("How do I reset the filter timer after replacing a filter?", MANUAL, "FILTER button for 5 seconds", ["5 seconds"]),
    ("How long is the warranty if I register the purifier?", MANUAL, "extended to 5 years", ["5 years"]),
    ("What is the tank capacity of the AquaPure X2?", MANUAL, "Tank capacity", ["6 litres"]),
    ("What are the four functions of the AI RMF Core?", NIST, "GOVERN, MAP, MEASURE, and MANAGE", ["govern", "map", "measure", "manage"]),
    ("Which characteristics make an AI system trustworthy according to the AI RMF?", NIST, "valid and reliable, safe, secure", ["valid", "reliable", "safe"]),
]

UNANSWERABLE = [
    "What is the capital of Australia?",
    "Who won the 2022 FIFA World Cup?",
    "How much does the AquaPure X2 cost?",
    "What is the dress code at Northwind Labs?",
    "How do I bake sourdough bread?",
    "What was Northwind Labs' revenue in 2025?",
]


def evidence_pages(store: VectorStore, filename: str, evidence: str) -> set[int]:
    """Pages of ``filename`` whose passages contain the evidence phrase."""
    norm = lambda text: " ".join(text.lower().split())  # noqa: E731
    pages = {c.page for c in store.chunks.values() if c.filename == filename and norm(evidence) in norm(c.text)}
    if not pages:
        raise ValueError(f"Evidence phrase {evidence!r} not found in {filename}")
    return pages


async def run(retrieval_only: bool, top_k: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(Path(tmp))
        client = OllamaClient(settings.ollama_base_url, settings.llm_model, settings.embedding_model, settings.request_timeout)
        rag = RAGService(store, client, dataclasses.replace(settings, data_dir=Path(tmp)))
        try:
            for pdf in sorted((ROOT / "samples").glob("*.pdf")):
                record = await rag.ingest_pdf(pdf.name, pdf.read_bytes())
                print(f"Indexed {record.filename}: {record.pages} pages, {record.chunks} passages")

            rows = []
            for question, document, evidence, keywords in ANSWERABLE:
                rows.append(await evaluate(rag, question, top_k, retrieval_only, True, document, evidence, keywords))
            for question in UNANSWERABLE:
                rows.append(await evaluate(rag, question, top_k, retrieval_only, False))
        finally:
            await client.aclose()
    return {"top_k": top_k, "min_similarity": settings.min_similarity, "llm": settings.llm_model,
            "embedding_model": settings.embedding_model, "rows": rows}


async def evaluate(rag: RAGService, question: str, top_k: int, retrieval_only: bool, answerable: bool,
                   document: str | None = None, evidence: str | None = None, keywords: list[str] | None = None) -> dict:
    vector = (await rag.client.embed([QUERY_PREFIX + question]))[0]
    hits = rag.store.hybrid_search(question, vector, top_k)
    row: dict = {
        "question": question,
        "answerable": answerable,
        "top_score": round(max((h.score for h in hits), default=0.0), 3),
        "retrieved": [f"{h.chunk.filename} p.{h.chunk.page} ({h.score:.2f})" for h in hits],
    }
    if answerable:
        pages = evidence_pages(rag.store, document, evidence)
        row["expected"] = f"{document} p.{', '.join(map(str, sorted(pages)))}"
        row["hit"] = any(h.chunk.filename == document and h.chunk.page in pages and rag.is_relevant(h) for h in hits)
    if not retrieval_only:
        result = await rag.answer(question, top_k)
        row["status"] = result.status
        row["answer"] = result.answer
        if answerable:
            row["correct"] = result.status == "answered" and all(k.lower() in result.answer.lower() for k in keywords)
        else:
            row["correct"] = result.status == "insufficient_context"
    print(f"[{'A' if answerable else 'U'}] {question}\n    top={row['top_score']} hit={row.get('hit', '-')} "
          f"status={row.get('status', '-')} correct={row.get('correct', '-')}")
    return row


def write_report(results: dict, retrieval_only: bool) -> None:
    rows = results["rows"]
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    a_scores = [r["top_score"] for r in answerable]
    u_scores = [r["top_score"] for r in unanswerable]
    lines = [
        "# RAG evaluation",
        "",
        f"Knowledge base: the PDFs in `samples/`. LLM: `{results['llm']}`, embeddings: `{results['embedding_model']}`, "
        f"top-k = {results['top_k']}, similarity threshold = {results['min_similarity']}.",
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Retrieval hit rate (answer page retrieved and judged relevant) | "
        f"{sum(r['hit'] for r in answerable)}/{len(answerable)} |",
    ]
    if not retrieval_only:
        lines += [
            f"| Answer accuracy (answered and mentions the expected facts) | {sum(r['correct'] for r in answerable)}/{len(answerable)} |",
            f"| Safe refusals for questions not covered by the documents | {sum(r['correct'] for r in unanswerable)}/{len(unanswerable)} |",
        ]
    lines += [
        f"| Top similarity, answerable questions (min / median / max) | {min(a_scores):.2f} / {statistics.median(a_scores):.2f} / {max(a_scores):.2f} |",
        f"| Top similarity, unanswerable questions (min / median / max) | {min(u_scores):.2f} / {statistics.median(u_scores):.2f} / {max(u_scores):.2f} |",
        "",
        "## Details",
        "",
        "| Type | Question | Top score | Retrieval hit | Status | Correct |",
        "|---|---|---:|:---:|---|:---:|",
    ]
    for r in rows:
        lines.append(
            f"| {'answerable' if r['answerable'] else 'not in docs'} | {r['question']} | {r['top_score']:.2f} | "
            f"{'yes' if r.get('hit') else ('no' if r['answerable'] else '-')} | {r.get('status', '-')} | "
            f"{'yes' if r.get('correct') else ('no' if 'correct' in r else '-')} |"
        )
    if not retrieval_only:
        lines += ["", "## Answers", ""]
        for r in rows:
            answer = r["answer"].replace("\n", "\n> ")
            lines += [f"**Q: {r['question']}**", "", f"> {answer}", ""]
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "rag_evaluation.md").write_text("\n".join(lines) + "\n")
    (out / "rag_evaluation.json").write_text(json.dumps(results, indent=2))
    print(f"\nReport written to {out / 'rag_evaluation.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval-only", action="store_true", help="skip answer generation (fast)")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    args = parser.parse_args()
    results = asyncio.run(run(args.retrieval_only, args.top_k))
    write_report(results, args.retrieval_only)


if __name__ == "__main__":
    main()
