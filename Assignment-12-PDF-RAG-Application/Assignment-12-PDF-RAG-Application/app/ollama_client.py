"""Small async client for the local Ollama REST API (chat + embeddings).

Every failure is translated into an ``OllamaError`` subclass that carries an HTTP
status code and a message that can be shown to the end user as-is.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    status_code = 502
    code = "model_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class OllamaUnavailableError(OllamaError):
    status_code = 503
    code = "ollama_unavailable"


class ModelNotFoundError(OllamaError):
    status_code = 503
    code = "model_not_found"


class OllamaTimeoutError(OllamaError):
    status_code = 504
    code = "model_timeout"


class InvalidModelOutputError(OllamaError):
    status_code = 502
    code = "invalid_model_output"


class OllamaClient:
    def __init__(self, base_url: str, llm_model: str, embedding_model: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(timeout, connect=5.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[str]:
        response = await self._request("GET", "/api/tags", model=None)
        return [m.get("name", "") for m in response.json().get("models", [])]

    async def missing_models(self) -> list[str]:
        installed = set(await self.list_models())
        return [m for m in (self.llm_model, self.embedding_model) if m not in installed and f"{m}:latest" not in installed]

    async def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed texts with the embedding model (batched)."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = await self._request(
                "POST", "/api/embed", model=self.embedding_model, json={"model": self.embedding_model, "input": batch}
            )
            embeddings = response.json().get("embeddings") or []
            if len(embeddings) != len(batch):
                raise InvalidModelOutputError("The embedding model returned an unexpected number of vectors.")
            vectors.extend(embeddings)
        return vectors

    async def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.1, max_tokens: int = 700) -> str:
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": max_tokens},
        }
        response = await self._request("POST", "/api/chat", model=self.llm_model, json=payload)
        content = (response.json().get("message") or {}).get("content", "").strip()
        if not content:
            raise InvalidModelOutputError("The model returned an empty answer. Please try again.")
        return content

    async def _request(self, method: str, path: str, *, model: str | None, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(
                f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is installed and running (`ollama serve`)."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError("The model took too long to respond. Please try again.") from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(f"Error while communicating with Ollama: {exc}") from exc

        if response.status_code == 404 and model:
            raise ModelNotFoundError(f"The model '{model}' is not installed in Ollama. Run `ollama pull {model}` and try again.")
        if response.status_code >= 400:
            try:
                detail = str(response.json().get("error") or response.text)
            except ValueError:
                detail = response.text or "unknown error"
            logger.error("Ollama %s %s failed with %s: %s", method, path, response.status_code, detail)
            raise OllamaError(f"Ollama returned an error ({response.status_code}): {detail}")
        return response
