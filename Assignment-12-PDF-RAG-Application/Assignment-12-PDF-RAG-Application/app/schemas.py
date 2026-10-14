"""Request and response models for the RAG API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import settings


class AskRequest(BaseModel):
    question: str = Field(..., description="The question to answer from the uploaded documents.")
    top_k: int | None = Field(None, description="How many passages to retrieve (default from settings).")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = " ".join(value.split())
        if not question:
            raise ValueError("Please enter a question.")
        if len(question) < 3:
            raise ValueError("The question is too short.")
        if len(question) > settings.max_question_chars:
            raise ValueError(f"The question is too long (maximum {settings.max_question_chars} characters).")
        return question

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int | None) -> int | None:
        if value is not None and not 1 <= value <= settings.max_top_k:
            raise ValueError(f"top_k must be between 1 and {settings.max_top_k}.")
        return value


class SourceModel(BaseModel):
    rank: int
    document_id: str
    filename: str
    page: int
    score: float = Field(..., description="Cosine similarity between the question and the passage.")
    text: str
    relevant: bool = Field(..., description="Whether the passage passed the relevance check and was sent to the LLM.")
    keyword_match: list[str] = Field(default_factory=list, description="Rare query terms (codes, names) found in the passage.")
    cited: bool = Field(..., description="Whether the answer cites this passage.")


class AskResponse(BaseModel):
    question: str
    answer: str
    status: Literal["answered", "insufficient_context", "no_documents"]
    sources: list[SourceModel]
    model: str
    retrieval_ms: int
    generation_ms: int


class DocumentModel(BaseModel):
    id: str
    filename: str
    pages: int
    chunks: int
    words: int
    size_bytes: int
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentModel]
    total_chunks: int


class UploadResult(BaseModel):
    filename: str
    status: Literal["indexed", "duplicate", "error"]
    message: str
    document: DocumentModel | None = None


class UploadResponse(BaseModel):
    results: list[UploadResult]


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    llm_model: str
    embedding_model: str
    missing_models: list[str]
    message: str


class ConfigResponse(BaseModel):
    llm_model: str
    embedding_model: str
    max_file_mb: int
    max_files_per_upload: int
    top_k: int
    max_top_k: int
    min_similarity: float
    max_question_chars: int


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
