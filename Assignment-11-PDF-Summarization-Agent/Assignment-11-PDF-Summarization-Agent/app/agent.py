"""The PDF summarization agent.

The agent turns an extracted PDF into a structured summary in explicit steps:

1. **Analyse** the document (pages, words, estimated tokens).
2. **Plan** a strategy: documents that fit comfortably in Llama 3's context are
   summarised in a single pass; longer ones use map-reduce.
3. **Map**: each chunk (a page-aware text segment) is condensed into notes.
4. **Reduce**: if the combined notes are still too long they are merged in
   groups, repeatedly, until they fit in the context budget.
5. **Synthesise**: the final summary is generated as JSON that follows a fixed
   schema (title, TL;DR, summary, key points) using Ollama structured outputs.
6. **Check**: the output is validated and normalised before it is returned.

Progress events are reported through a callback so the web UI can show them.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable

from .chunker import Chunk, chunk_pages, estimate_tokens
from .config import Settings
from .ollama_client import InvalidModelOutputError, OllamaClient
from .pdf_extractor import ExtractedDocument

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a meticulous document analyst. You summarise documents faithfully, using only "
    "information that appears in the provided text. You never invent facts, numbers or names."
)


class SummaryLength(str, Enum):
    brief = "brief"
    standard = "standard"
    detailed = "detailed"


@dataclass(frozen=True)
class LengthSpec:
    summary_instruction: str
    key_points: int
    max_tokens: int


LENGTH_SPECS: dict[SummaryLength, LengthSpec] = {
    SummaryLength.brief: LengthSpec("a single paragraph of 3 to 4 sentences (no line breaks)", 4, 700),
    SummaryLength.standard: LengthSpec(
        "two paragraphs, about 150 to 220 words in total, separated by a blank line (\\n\\n)", 6, 1000
    ),
    SummaryLength.detailed: LengthSpec(
        "three or four paragraphs, about 300 to 400 words in total, separated by a blank line (\\n\\n)", 8, 1400
    ),
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "tldr": {"type": "string"},
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "tldr", "summary", "key_points"],
}


@dataclass
class ProgressEvent:
    stage: str  # analyzing | planning | mapping | reducing | writing | done
    message: str
    current: int | None = None
    total: int | None = None


ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]


async def _no_progress(_: ProgressEvent) -> None:
    return None


@dataclass
class SectionNotes:
    start_page: int
    end_page: int
    notes: list[str]


@dataclass
class AgentResult:
    title: str
    tldr: str
    summary: str
    key_points: list[str]
    sections: list[SectionNotes] = field(default_factory=list)
    strategy: str = "single-pass"
    chunks: int = 1
    llm_calls: int = 0
    duration_ms: int = 0


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #
def build_map_messages(chunk: Chunk, total_chunks: int, filename: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"The text below is part {chunk.index + 1} of {total_chunks} of the document "
                f"\"{filename}\" ({chunk.page_label}).\n\n"
                f"<document_part>\n{chunk.text}\n</document_part>\n\n"
                "Write concise notes on this part of the document as 4 to 8 bullet points that start with \"- \". "
                "Capture the main ideas, important facts, figures, names, decisions and conclusions. "
                "Only include information that appears in the text. Reply with the bullet points only."
            ),
        },
    ]


def build_reduce_messages(notes_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Below are notes taken from consecutive sections of one document.\n\n"
                f"<notes>\n{notes_text}\n</notes>\n\n"
                "Merge them into a single, shorter set of 6 to 10 bullet points that start with \"- \". "
                "Remove repetition, keep the most important facts and figures, and keep the original order of ideas. "
                "Reply with the bullet points only."
            ),
        },
    ]


def build_final_messages(
    content: str, *, is_notes: bool, length: SummaryLength, document: ExtractedDocument
) -> list[dict[str, str]]:
    spec = LENGTH_SPECS[length]
    source = "notes covering the whole document" if is_notes else "full text of the document"
    title_hint = f" The PDF metadata title is \"{document.title}\"." if document.title else ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Below is the {source} \"{document.filename}\" ({document.page_count} pages).{title_hint}\n\n"
                f"<content>\n{content}\n</content>\n\n"
                "Summarise the whole document and answer with a JSON object containing:\n"
                "- \"title\": a short, descriptive title for the document (at most 12 words)\n"
                "- \"tldr\": one sentence that captures the main message\n"
                f"- \"summary\": {spec.summary_instruction}. "
                "It must cover the document from beginning to end, including its later sections and conclusions, "
                "not only the first part\n"
                f"- \"key_points\": a list of {spec.key_points} key points drawn from all parts of the document, "
                "each one complete sentence\n"
                "Use only information from the content above (do not add facts, places or numbers that it does not "
                "mention) and write in plain, clear English."
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_PREAMBLE = re.compile(r"^\s*(here (are|is)|below (are|is)|sure|certainly|notes?:)", re.IGNORECASE)


def parse_bullets(text: str) -> list[str]:
    """Extract bullet points from model output (tolerates numbering and preambles)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullets = [_BULLET.sub("", line).strip() for line in lines if _BULLET.match(line)]
    if bullets:
        return [b for b in bullets if b]
    # No bullet markers: fall back to non-preamble lines.
    return [line for line in lines if not (_PREAMBLE.match(line) and line.endswith(":"))]


def _clean_str(value: object) -> str:
    return re.sub(r"[ \t]+", " ", str(value or "")).strip()


def normalise_final(data: dict, length: SummaryLength, fallback_title: str) -> tuple[str, str, str, list[str]]:
    title = _clean_str(data.get("title")).strip("\"'") or fallback_title
    tldr = _clean_str(data.get("tldr"))
    summary = "\n\n".join(
        _clean_str(p) for p in re.split(r"\s*\n\s*|\\n", str(data.get("summary") or "")) if _clean_str(p)
    )
    raw_points = data.get("key_points") or []
    if isinstance(raw_points, str):
        raw_points = parse_bullets(raw_points)
    points: list[str] = []
    for point in raw_points:
        text = _BULLET.sub("", _clean_str(point))
        if text and text.lower() not in {p.lower() for p in points}:
            points.append(text)
    points = points[: LENGTH_SPECS[length].key_points + 2]

    if not summary or not points:
        raise InvalidModelOutputError("The model returned an incomplete summary. Please try again.")
    if not tldr:
        tldr = re.split(r"(?<=[.!?])\s", summary, maxsplit=1)[0]
    return title, tldr, summary, points


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class PDFSummarizationAgent:
    def __init__(self, client: OllamaClient, settings: Settings):
        self.client = client
        self.settings = settings
        self.llm_calls = 0

    async def _chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.llm_calls += 1
        result = await self.client.chat(messages, **kwargs)
        return result.content

    async def _summarise_chunk(self, chunk: Chunk, total: int, filename: str) -> SectionNotes:
        content = await self._chat(build_map_messages(chunk, total, filename), temperature=0.1, max_tokens=450)
        notes = parse_bullets(content) or [content]
        return SectionNotes(start_page=chunk.start_page, end_page=chunk.end_page, notes=notes)

    async def _reduce(self, notes_blocks: list[str], progress: ProgressCallback) -> str:
        """Merge note blocks group by group until they fit in the notes budget."""
        budget = self.settings.notes_budget_tokens
        round_no = 0
        while estimate_tokens("\n\n".join(notes_blocks)) > budget and len(notes_blocks) > 1:
            round_no += 1
            groups: list[list[str]] = [[]]
            for block in notes_blocks:
                if groups[-1] and estimate_tokens("\n\n".join(groups[-1] + [block])) > budget:
                    groups.append([])
                groups[-1].append(block)
            if len(groups) == len(notes_blocks):  # every block is already as big as the budget
                groups = [notes_blocks[i:i + 2] for i in range(0, len(notes_blocks), 2)]
            merged: list[str] = []
            for i, group in enumerate(groups, start=1):
                await progress(ProgressEvent("reducing", f"Merging notes (round {round_no}, group {i} of {len(groups)})", i, len(groups)))
                content = await self._chat(build_reduce_messages("\n\n".join(group)), temperature=0.1, max_tokens=700)
                merged.append("\n".join(f"- {b}" for b in parse_bullets(content)) or content)
            notes_blocks = merged
        return "\n\n".join(notes_blocks)

    async def run(
        self,
        document: ExtractedDocument,
        length: SummaryLength = SummaryLength.standard,
        progress: ProgressCallback = _no_progress,
    ) -> AgentResult:
        started = time.perf_counter()
        self.llm_calls = 0

        # 1. Analyse
        full_text = document.text
        tokens = estimate_tokens(full_text)
        await progress(
            ProgressEvent(
                "analyzing",
                f"Read {document.page_count} pages, {document.word_count:,} words (about {tokens:,} tokens).",
            )
        )

        # 2. Plan
        sections: list[SectionNotes] = []
        if tokens <= self.settings.single_pass_tokens:
            strategy, chunks_count = "single-pass", 1
            await progress(ProgressEvent("planning", "The document fits in the model's context: summarising in a single pass."))
            final_input, is_notes = full_text, False
        else:
            chunks = chunk_pages(document.pages, self.settings.chunk_tokens, self.settings.chunk_overlap_tokens)
            strategy, chunks_count = "map-reduce", len(chunks)
            await progress(
                ProgressEvent("planning", f"Long document: split into {len(chunks)} overlapping sections for map-reduce summarisation.")
            )

            # 3. Map
            for chunk in chunks:
                await progress(
                    ProgressEvent(
                        "mapping",
                        f"Summarising section {chunk.index + 1} of {len(chunks)} ({chunk.page_label})",
                        chunk.index + 1,
                        len(chunks),
                    )
                )
                sections.append(await self._summarise_chunk(chunk, len(chunks), document.filename))

            # 4. Reduce
            blocks = [
                f"[{'Page' if s.start_page == s.end_page else 'Pages'} "
                f"{s.start_page}{'' if s.start_page == s.end_page else f'-{s.end_page}'}]\n"
                + "\n".join(f"- {n}" for n in s.notes)
                for s in sections
            ]
            final_input, is_notes = await self._reduce(blocks, progress), True

        # 5. Synthesise
        await progress(ProgressEvent("writing", "Writing the final summary with Llama 3…"))
        spec = LENGTH_SPECS[length]
        self.llm_calls += 1
        data = await self.client.chat_json(
            build_final_messages(final_input, is_notes=is_notes, length=length, document=document),
            FINAL_SCHEMA,
            temperature=0.2,
            max_tokens=spec.max_tokens,
        )

        # 6. Check
        fallback_title = document.title or document.filename.rsplit(".", 1)[0]
        title, tldr, summary, key_points = normalise_final(data, length, fallback_title)
        duration_ms = int((time.perf_counter() - started) * 1000)
        await progress(ProgressEvent("done", f"Done in {duration_ms / 1000:.1f}s."))
        logger.info(
            "Summarised %s (%d pages, %s strategy, %d chunks, %d LLM calls) in %d ms",
            document.filename, document.page_count, strategy, chunks_count, self.llm_calls, duration_ms,
        )
        return AgentResult(
            title=title,
            tldr=tldr,
            summary=summary,
            key_points=key_points,
            sections=sections,
            strategy=strategy,
            chunks=chunks_count,
            llm_calls=self.llm_calls,
            duration_ms=duration_ms,
        )
