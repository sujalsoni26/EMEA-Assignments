"""Summarization logic: prompt construction, chunking and output clean-up.

TinyLlama (1.1B parameters) is small, so this module does some extra work around
the model to get dependable results:

* The source text is placed *before* the instruction, which small chat models
  follow more reliably than an instruction followed by a long text.
* TinyLlama has a 2,048-token context window. Long inputs are split into chunks,
  each chunk is summarised, and the partial summaries are summarised again
  (a simple map-reduce strategy).
* The model tends to ignore requested lengths, so the output is cleaned
  (preambles, echoed instructions, unfinished sentences, duplicate bullets) and
  then trimmed to the requested length by keeping the most informative
  sentences in their original order.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from .ollama_client import EmptyResponseError, OllamaClient
from .schemas import SummaryFormat, SummaryLength, count_words

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that summarizes text accurately and concisely. "
    "You only use facts from the given text and never add opinions or extra information."
)


@dataclass(frozen=True)
class LengthPreset:
    paragraph_instruction: str
    bullet_instruction: str
    max_sentences: int
    max_bullets: int
    num_predict: int


LENGTH_PRESETS: dict[SummaryLength, LengthPreset] = {
    SummaryLength.short: LengthPreset(
        paragraph_instruction="Write a very short summary of the text above in one or two sentences.",
        bullet_instruction="List the 3 most important points of the text above as bullet points starting with \"- \".",
        max_sentences=2,
        max_bullets=3,
        num_predict=220,
    ),
    SummaryLength.medium: LengthPreset(
        paragraph_instruction="Write a concise summary of the text above in one paragraph of about four sentences.",
        bullet_instruction="List the 5 most important points of the text above as bullet points starting with \"- \".",
        max_sentences=4,
        max_bullets=5,
        num_predict=300,
    ),
    SummaryLength.long: LengthPreset(
        paragraph_instruction="Write a detailed summary of the text above in one paragraph that covers all the main points.",
        bullet_instruction="List the 7 most important points of the text above as bullet points starting with \"- \".",
        max_sentences=7,
        max_bullets=7,
        num_predict=420,
    ),
}

MAX_REDUCE_ROUNDS = 3

GENERATION_OPTIONS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_ctx": 2048,
}

_STOPWORDS = set(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me more most
    my myself no nor not now of off on once only or other our ours ourselves out over own same she should
    so some such than that the their theirs them themselves then there these they this those through to
    too under until up very was we were what when where which while who whom why will with would you your
    yours yourself yourselves also text article summary discusses describes""".split()
)


@dataclass
class SummaryResult:
    summary: str
    chunks: int
    duration_ms: int


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def split_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most ``max_chars`` characters.

    Paragraph boundaries are preferred, then sentence boundaries, and only as a
    last resort are very long sentences split on word boundaries.
    """
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        for sentence in split_sentences(paragraph):
            if len(sentence) <= max_chars:
                pieces.append(sentence)
            else:
                pieces.extend(_split_words(sentence, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _split_words(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = word[:max_chars]
    if current:
        parts.append(current)
    return parts


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #
def build_messages(text: str, length: SummaryLength, fmt: SummaryFormat) -> list[dict[str, str]]:
    preset = LENGTH_PRESETS[length]
    if fmt == SummaryFormat.bullets:
        instruction = f"{preset.bullet_instruction} Reply with the bullet points only."
    else:
        instruction = f"{preset.paragraph_instruction} Reply with the summary only."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{text}\n\n---\n{instruction}"},
    ]


# --------------------------------------------------------------------------- #
# Output clean-up
# --------------------------------------------------------------------------- #
_PREAMBLE_LINE = re.compile(
    r"^\s*(sure|certainly|okay|ok|of course|absolutely)\b.*$"
    r"|^\s*(here\s+(is|are|'s)|below\s+(is|are))\b.*$"
    r"|^\s*(summary|bullet points|key points|main points|tl;?dr)\s*:?\s*$"
    r"|^\s*(write|list|summari[sz]e|give)\b.*\b(text|points|summary)\b.*$",
    re.IGNORECASE,
)
# A preamble glued to the start of the real content, e.g. "Sure! Here's a summary: ..."
_INLINE_PREFIX = re.compile(
    r"^\s*(?:(?:sure|certainly|okay|ok|of course|absolutely)\b[!,.]?\s*)?"
    r"(?:(?:here\s+(?:is|are|'s)|below\s+(?:is|are))\b[^:\n]{0,60}:\s*)?"
    r"(?:(?:summary|tl;?dr)\s*:\s*)?",
    re.IGNORECASE,
)
_CLOSING_LINE = re.compile(
    r"^\s*(i hope|let me know|feel free|in summary,? (this|the) (summary|text)|note:)\b.*$",
    re.IGNORECASE,
)
_BULLET_MARKER = re.compile(r"^\s*(?:[-*•‣◦]|\d+[.)])\s+")


def _is_preamble(line: str) -> bool:
    # Only short lines or lines ending in ":" are treated as preambles, so a
    # one-line answer such as "Sure! Coral reefs ..." is never thrown away.
    stripped = line.strip()
    return bool(_PREAMBLE_LINE.match(stripped)) and (stripped.endswith(":") or len(stripped) <= 60)


def _strip_noise(raw: str) -> list[str]:
    lines = [line.rstrip() for line in raw.replace("\r\n", "\n").split("\n")]
    cleaned = [line for line in lines if line.strip()]
    while cleaned and _is_preamble(cleaned[0]):
        cleaned.pop(0)
    while cleaned and _CLOSING_LINE.match(cleaned[-1]):
        cleaned.pop()
    if cleaned:
        cleaned[0] = _INLINE_PREFIX.sub("", cleaned[0])
    return [line.strip() for line in cleaned if line.strip()]


def _is_complete(sentence: str) -> bool:
    return bool(re.search(r"[.!?][\"')\]]?$", sentence.strip()))


def _normalize(word: str) -> str:
    """Very light stemming so that "coral" and "corals" count as the same word."""
    word = word.removesuffix("'s")
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _content_words(text: str) -> list[str]:
    return [
        _normalize(w)
        for w in re.findall(r"[a-z0-9']+", text.lower())
        if w not in _STOPWORDS and len(w) > 2
    ]


def _similar(a: str, b: str, threshold: float = 0.6) -> bool:
    wa, wb = set(_content_words(a)), set(_content_words(b))
    if not wa or not wb:
        return a.strip().lower() == b.strip().lower()
    return len(wa & wb) / min(len(wa), len(wb)) >= threshold


def _dedupe(items: list[str]) -> list[str]:
    kept: list[str] = []
    for item in items:
        if not any(_similar(item, existing) for existing in kept):
            kept.append(item)
    return kept


def select_informative(sentences: list[str], limit: int, source_text: str) -> list[str]:
    """Keep at most ``limit`` sentences using a SumBasic-style selection.

    Sentences are scored by the average source frequency of their content words.
    After a sentence is picked, the weight of its words is squared (reduced),
    so later picks favour sentences that cover *new* information. The first
    sentence (usually the topic sentence) is always kept, and the original order
    is preserved.
    """
    if len(sentences) <= limit:
        return sentences
    counts: dict[str, int] = {}
    for word in _content_words(source_text):
        counts[word] = counts.get(word, 0) + 1
    total = sum(counts.values()) or 1
    prob = {word: count / total for word, count in counts.items()}

    def score(sentence: str) -> float:
        words = set(_content_words(sentence))
        return sum(prob.get(w, 0.0) for w in words) / len(words) if words else 0.0

    def mark_used(sentence: str) -> None:
        for word in set(_content_words(sentence)):
            if word in prob:
                prob[word] = prob[word] ** 2

    chosen = [0]
    mark_used(sentences[0])
    while len(chosen) < limit:
        candidates = [i for i in range(len(sentences)) if i not in chosen]
        best = max(candidates, key=lambda i: score(sentences[i]))
        chosen.append(best)
        mark_used(sentences[best])
    return [sentences[i] for i in sorted(chosen)]


def _damerau_distance_is_one(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` differ by exactly one insertion, deletion,
    substitution or swap of two adjacent letters."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        return len(diffs) == 2 and diffs[1] == diffs[0] + 1 and a[diffs[0]] == b[diffs[1]] and a[diffs[1]] == b[diffs[0]]
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(long_)):
        if long_[:i] + long_[i + 1:] == short:
            return True
    return False


def fix_typos_against_source(summary: str, source_text: str) -> str:
    """Correct obvious misspellings by snapping them to a word from the source.

    Small models occasionally garble words they copy from the input (TinyLlama
    for example often writes "ocena" for "ocean"). A summary word is replaced
    only when it does not occur in the source, is at least 5 letters long, is
    exactly one edit away from a single source word, and the difference is not
    just an inflection at the end of the word (e.g. "expert" vs "experts").
    """
    source_words = set(re.findall(r"[A-Za-z]+", source_text.lower()))
    by_length: dict[int, list[str]] = {}
    for word in source_words:
        if len(word) >= 4:
            by_length.setdefault(len(word), []).append(word)

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        lower = word.lower()
        if len(lower) < 5 or lower in source_words:
            return word
        candidates = [
            cand
            for size in (len(lower) - 1, len(lower), len(lower) + 1)
            for cand in by_length.get(size, [])
            if _damerau_distance_is_one(lower, cand)
        ]
        if len(candidates) != 1:
            return word
        candidate = candidates[0]
        shared_prefix = len(_common_prefix(lower, candidate))
        if shared_prefix >= min(len(lower), len(candidate)) - 1:
            return word  # differs only at the end: probably an inflection, not a typo
        if word.isupper():
            return candidate.upper()
        if word[0].isupper():
            return candidate.capitalize()
        return candidate

    return re.sub(r"[A-Za-z]+", replace, summary)


def _common_prefix(a: str, b: str) -> str:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[:i]


def clean_summary(
    raw: str, fmt: SummaryFormat, length: SummaryLength, source_text: str, truncated: bool = False
) -> str:
    """Turn raw model output into a clean summary of the requested format and length.

    ``truncated`` tells whether generation stopped at the token limit, in which
    case an unfinished last sentence is removed.
    """
    preset = LENGTH_PRESETS[length]
    lines = _strip_noise(raw)
    if not lines:
        return ""

    if fmt == SummaryFormat.bullets:
        bullet_lines = [line for line in lines if _BULLET_MARKER.match(line)]
        if len(bullet_lines) >= 2:
            items = [_BULLET_MARKER.sub("", line).strip() for line in bullet_lines]
        else:
            # The model answered with prose: turn its sentences into bullets.
            items = split_sentences(" ".join(lines))
        # One idea per bullet: split bullets that contain several sentences.
        items = [sentence for item in items if item for sentence in (split_sentences(item) or [item])]
        if truncated and len(items) > 1 and not _is_complete(items[-1]):
            items.pop()  # generation hit the token limit mid-sentence
        items = [item if _is_complete(item) else f"{item}." for item in items]
        items = _dedupe(items)
        items = select_informative(items, preset.max_bullets, source_text)
        return fix_typos_against_source("\n".join(f"- {item}" for item in items), source_text)

    prose = " ".join(_BULLET_MARKER.sub("", line) for line in lines)
    sentences = split_sentences(prose)
    if truncated and len(sentences) > 1 and not _is_complete(sentences[-1]):
        sentences.pop()  # generation hit the token limit mid-sentence
    sentences = _dedupe(sentences)
    sentences = select_informative(sentences, preset.max_sentences, source_text)
    summary = " ".join(sentences).strip()
    if summary and not _is_complete(summary):
        summary += "."
    return fix_typos_against_source(summary, source_text)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
class Summarizer:
    def __init__(self, client: OllamaClient, chunk_chars: int):
        self.client = client
        self.chunk_chars = chunk_chars

    async def _summarize_once(self, text: str, length: SummaryLength, fmt: SummaryFormat) -> str:
        preset = LENGTH_PRESETS[length]
        options = {**GENERATION_OPTIONS, "num_predict": preset.num_predict}
        result = await self.client.chat(build_messages(text, length, fmt), options)
        cleaned = clean_summary(result.content, fmt, length, text, truncated=result.truncated)
        if not cleaned:
            raise EmptyResponseError("The model did not produce a usable summary. Please try again.")
        return cleaned

    async def summarize(self, text: str, length: SummaryLength, fmt: SummaryFormat) -> SummaryResult:
        started = time.perf_counter()
        chunks = split_into_chunks(text, self.chunk_chars)
        total_chunks = len(chunks)

        # Map-reduce: summarise each chunk until the combined text fits in one call.
        rounds = 0
        while len(chunks) > 1 and rounds < MAX_REDUCE_ROUNDS:
            rounds += 1
            logger.info("Reduce round %d: summarising %d chunks", rounds, len(chunks))
            partials = [
                await self._summarize_once(chunk, SummaryLength.medium, SummaryFormat.paragraph)
                for chunk in chunks
            ]
            chunks = split_into_chunks("\n\n".join(partials), self.chunk_chars)

        # Safety net: if the partial summaries still do not fit, keep what fits.
        final_input = chunks[0] if len(chunks) == 1 else "\n\n".join(chunks)[: self.chunk_chars]
        summary = await self._summarize_once(final_input, length, fmt)
        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Summarised %d words into %d words in %d ms (%d chunks)",
            count_words(text), count_words(summary), duration_ms, total_chunks,
        )
        return SummaryResult(summary=summary, chunks=total_chunks, duration_ms=duration_ms)
