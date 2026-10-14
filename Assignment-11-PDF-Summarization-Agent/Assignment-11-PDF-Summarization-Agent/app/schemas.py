"""Response models for the PDF summarization API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .agent import SummaryLength


class SectionNotesModel(BaseModel):
    start_page: int
    end_page: int
    notes: list[str]


class DocumentInfo(BaseModel):
    filename: str
    pages: int
    words: int
    title: str | None = None
    skipped_pages: list[int] = Field(default_factory=list)


class SummaryStats(BaseModel):
    model: str
    strategy: str = Field(..., description="'single-pass' or 'map-reduce'")
    chunks: int
    llm_calls: int
    extraction_ms: int
    summarization_ms: int


class SummaryResponse(BaseModel):
    title: str
    tldr: str
    summary: str
    key_points: list[str]
    sections: list[SectionNotesModel]
    length: SummaryLength
    document: DocumentInfo
    stats: SummaryStats


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    model: str
    model_available: bool
    message: str


class ConfigResponse(BaseModel):
    model: str
    max_file_mb: int
    max_pages: int


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
