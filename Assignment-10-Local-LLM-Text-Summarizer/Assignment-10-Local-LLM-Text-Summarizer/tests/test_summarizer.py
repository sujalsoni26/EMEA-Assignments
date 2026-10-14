"""Unit tests for chunking and output clean-up."""

from __future__ import annotations

from app.schemas import SummaryFormat, SummaryLength
from app.summarizer import (
    build_messages,
    clean_summary,
    fix_typos_against_source,
    select_informative,
    split_into_chunks,
    split_sentences,
)

SOURCE = "Coral reefs support many species. Warm ocean water causes coral bleaching. Experts want lower emissions."


def test_short_text_is_a_single_chunk():
    assert split_into_chunks("Hello world.", 100) == ["Hello world."]


def test_chunks_respect_size_and_keep_all_words():
    paragraphs = [f"Paragraph {i}. " + "This is a sentence about the topic. " * 8 for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = split_into_chunks(text, 600)
    assert len(chunks) > 1
    assert all(len(chunk) <= 600 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_very_long_sentence_is_split_on_words():
    text = "word " * 500
    chunks = split_into_chunks(text, 200)
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert sum(len(c.split()) for c in chunks) == 500


def test_split_sentences():
    assert split_sentences("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]


def test_prompt_puts_text_before_instruction():
    messages = build_messages("SOURCE TEXT", SummaryLength.short, SummaryFormat.bullets)
    assert messages[0]["role"] == "system"
    assert messages[1]["content"].startswith("SOURCE TEXT")
    assert "bullet points" in messages[1]["content"]


def test_clean_summary_removes_preamble_and_cut_off_sentence():
    raw = "Sure, here is a summary of the text:\nCoral reefs support many species. Warm water causes bleaching. Experts"
    assert clean_summary(raw, SummaryFormat.paragraph, SummaryLength.long, SOURCE, truncated=True) == (
        "Coral reefs support many species. Warm water causes bleaching."
    )


def test_unfinished_sentence_is_kept_when_output_was_not_truncated():
    raw = "Coral reefs support many species. Warm water causes bleaching"
    assert clean_summary(raw, SummaryFormat.paragraph, SummaryLength.long, SOURCE) == (
        "Coral reefs support many species. Warm water causes bleaching."
    )


def test_clean_summary_keeps_one_line_answer_that_starts_with_sure():
    raw = "Sure! Coral reefs support many species and are threatened by warm water."
    assert clean_summary(raw, SummaryFormat.paragraph, SummaryLength.long, SOURCE) == (
        "Coral reefs support many species and are threatened by warm water."
    )


def test_clean_summary_strips_inline_summary_prefix():
    raw = "Summary: Coral reefs support many species."
    assert clean_summary(raw, SummaryFormat.paragraph, SummaryLength.long, SOURCE) == "Coral reefs support many species."


def test_clean_summary_normalises_numbered_bullets_and_removes_duplicates():
    raw = (
        "Here are the key points:\n"
        "1. Coral reefs support many species.\n"
        "2) Warm water causes coral bleaching.\n"
        "* Coral reefs support many marine species.\n"
        "- Experts want lower emissions"
    )
    assert clean_summary(raw, SummaryFormat.bullets, SummaryLength.long, SOURCE) == (
        "- Coral reefs support many species.\n- Warm water causes coral bleaching.\n- Experts want lower emissions."
    )


def test_prose_answer_is_converted_to_bullets():
    raw = "Coral reefs support many species. Warm water causes bleaching."
    assert clean_summary(raw, SummaryFormat.bullets, SummaryLength.medium, SOURCE) == (
        "- Coral reefs support many species.\n- Warm water causes bleaching."
    )


def test_summary_is_trimmed_to_requested_length():
    raw = (
        "Coral reefs support many species. Polyps live with algae. Warm ocean water causes coral bleaching. "
        "Reefs protect coastlines. Tourism earns billions. Experts want lower emissions."
    )
    summary = clean_summary(raw, SummaryFormat.paragraph, SummaryLength.short, SOURCE)
    assert len(split_sentences(summary)) == 2
    assert summary.startswith("Coral reefs support many species.")


def test_select_informative_keeps_first_sentence_and_order():
    sentences = ["Coral reefs matter.", "Unrelated filler text here.", "Warm ocean water causes coral bleaching."]
    assert select_informative(sentences, 2, SOURCE) == ["Coral reefs matter.", "Warm ocean water causes coral bleaching."]


def test_fix_typos_snaps_garbled_words_to_source():
    assert fix_typos_against_source("When ocena water warms.", SOURCE) == "When ocean water warms."
    assert fix_typos_against_source("Ocena levels.", SOURCE) == "Ocean levels."


def test_fix_typos_leaves_inflections_and_unknown_words_alone():
    text = "Expert opinion differs; zooxanthellae matter."
    assert fix_typos_against_source(text, SOURCE) == text
