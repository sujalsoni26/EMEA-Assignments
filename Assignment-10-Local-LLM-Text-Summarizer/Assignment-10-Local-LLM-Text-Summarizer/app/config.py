"""Application settings, read from environment variables with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {value!r}") from exc


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "tinyllama")
    # Seconds to wait for a single generation call before giving up.
    request_timeout: float = _env_float("REQUEST_TIMEOUT", 180.0)
    # Input limits (also shown to the user by the web interface).
    min_words: int = _env_int("MIN_INPUT_WORDS", 30)
    max_chars: int = _env_int("MAX_INPUT_CHARS", 12000)
    # TinyLlama has a 2048-token context window. Texts longer than this many
    # characters are split into chunks that are summarised separately first.
    chunk_chars: int = _env_int("CHUNK_CHARS", 3500)


settings = Settings()
