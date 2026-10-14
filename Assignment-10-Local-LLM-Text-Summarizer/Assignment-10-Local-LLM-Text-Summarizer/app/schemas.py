"""Request and response models for the summarization API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from .config import settings


class SummaryLength(str, Enum):
    short = "short"
    medium = "medium"
    long = "long"


class SummaryFormat(str, Enum):
    paragraph = "paragraph"
    bullets = "bullets"


def count_words(text: str) -> int:
    return len(text.split())


class SummarizeRequest(BaseModel):
    text: str = Field(..., description="The text to summarize.")
    length: SummaryLength = Field(SummaryLength.medium, description="How long the summary should be.")
    format: SummaryFormat = Field(SummaryFormat.paragraph, description="Paragraph or bullet points.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Please enter some text to summarize.")
        if len(text) > settings.max_chars:
            raise ValueError(
                f"The text is too long ({len(text):,} characters). "
                f"The maximum is {settings.max_chars:,} characters."
            )
        words = count_words(text)
        if words < settings.min_words:
            raise ValueError(
                f"The text is too short to summarize ({words} words). "
                f"Please provide at least {settings.min_words} words."
            )
        if not any(ch.isalpha() for ch in text):
            raise ValueError("The text does not contain any readable words.")
        return text

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Paste a paragraph or article of at least 30 words here...",
                    "length": "medium",
                    "format": "paragraph",
                }
            ]
        }
    }


class SummarizeResponse(BaseModel):
    summary: str
    format: SummaryFormat
    length: SummaryLength
    model: str
    input_words: int
    summary_words: int
    chunks: int = Field(..., description="Number of chunks the input was split into.")
    duration_ms: int


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    model: str
    model_available: bool
    message: str


class ConfigResponse(BaseModel):
    model: str
    min_words: int
    max_chars: int


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
