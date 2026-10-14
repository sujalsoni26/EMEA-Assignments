"""Application settings, read from environment variables with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")
    # Seconds to wait for a single LLM call.
    request_timeout: float = _env_float("REQUEST_TIMEOUT", 300.0)
    # Llama 3 (8B) supports an 8,192-token context window.
    context_tokens: int = _env_int("CONTEXT_TOKENS", 8192)

    # Upload limits
    max_file_mb: int = _env_int("MAX_FILE_MB", 20)
    max_pages: int = _env_int("MAX_PAGES", 300)
    min_words: int = _env_int("MIN_WORDS", 50)

    # Segmentation
    # Documents up to this many (estimated) tokens are summarised in a single pass.
    single_pass_tokens: int = _env_int("SINGLE_PASS_TOKENS", 5500)
    # Otherwise the text is split into chunks of about this many tokens...
    chunk_tokens: int = _env_int("CHUNK_TOKENS", 3000)
    # ...that overlap by this many tokens, so ideas are not cut in half.
    chunk_overlap_tokens: int = _env_int("CHUNK_OVERLAP_TOKENS", 150)
    # Notes are merged until they fit in this budget before the final summary.
    notes_budget_tokens: int = _env_int("NOTES_BUDGET_TOKENS", 4500)

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024


settings = Settings()
