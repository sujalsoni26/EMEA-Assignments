"""Application settings, read from environment variables with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "llama3")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    request_timeout: float = _env_float("REQUEST_TIMEOUT", 300.0)

    # Where the FAISS index and chunk metadata are stored.
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data"))))

    # Upload limits
    max_file_mb: int = _env_int("MAX_FILE_MB", 25)
    max_pages: int = _env_int("MAX_PAGES", 500)
    max_files_per_upload: int = _env_int("MAX_FILES_PER_UPLOAD", 10)

    # Chunking (characters). Chunks never span two pages, so citations are exact.
    chunk_chars: int = _env_int("CHUNK_CHARS", 1000)
    chunk_overlap_chars: int = _env_int("CHUNK_OVERLAP_CHARS", 150)

    # Retrieval
    top_k: int = _env_int("TOP_K", 4)
    max_top_k: int = _env_int("MAX_TOP_K", 10)
    # Chunks whose cosine similarity to the question is below this value are
    # considered irrelevant. Calibrated for nomic-embed-text (see README).
    min_similarity: float = _env_float("MIN_SIMILARITY", 0.55)

    # Question limits
    max_question_chars: int = _env_int("MAX_QUESTION_CHARS", 1000)

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024


settings = Settings()
