"""Small async client for the local Ollama REST API.

Every failure is translated into an ``OllamaError`` subclass that carries an HTTP
status code and a message that can be shown to the end user as-is.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Base class for errors raised while talking to Ollama."""

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


@dataclass(frozen=True)
class ChatResult:
    content: str
    truncated: bool = False


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float, context_tokens: int = 8192):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.context_tokens = context_tokens
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(timeout, connect=5.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[str]:
        response = await self._request("GET", "/api/tags")
        return [m.get("name", "") for m in response.json().get("models", [])]

    async def is_model_available(self) -> bool:
        wanted = {self.model, f"{self.model}:latest"}
        return any(name in wanted for name in await self.list_models())

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": self.context_tokens, "num_predict": max_tokens},
        }
        if json_schema is not None:
            payload["format"] = json_schema  # Ollama structured outputs
        response = await self._request("POST", "/api/chat", json=payload)
        try:
            data = response.json()
        except ValueError as exc:
            raise InvalidModelOutputError("Ollama returned a response that is not valid JSON.") from exc
        content = (data.get("message") or {}).get("content", "").strip()
        if not content:
            raise InvalidModelOutputError("The model returned an empty response. Please try again.")
        return ChatResult(content=content, truncated=data.get("done_reason") == "length")

    async def chat_json(
        self, messages: list[dict[str, str]], schema: dict[str, Any], *, max_tokens: int = 1024, **kwargs: Any
    ) -> dict[str, Any]:
        """Ask for output that follows ``schema`` and parse it, retrying once on bad JSON.

        If the first answer was cut off by the token limit, the retry gets a bigger budget.
        """
        last_error: Exception | None = None
        for attempt in range(2):
            result = await self.chat(messages, json_schema=schema, max_tokens=max_tokens, **kwargs)
            try:
                parsed = json.loads(result.content)
                if isinstance(parsed, dict):
                    return parsed
                last_error = ValueError("top-level JSON value is not an object")
            except json.JSONDecodeError as exc:
                last_error = exc
            logger.warning("Model returned invalid JSON (attempt %d): %s", attempt + 1, last_error)
            if result.truncated:
                max_tokens *= 2
        raise InvalidModelOutputError(
            "The model did not return a well-formed summary. Please try again."
        ) from last_error

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is installed and running (`ollama serve`)."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                "The model took too long to respond. Try again, or try a shorter document."
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(f"Error while communicating with Ollama: {exc}") from exc

        if response.status_code == 404:
            raise ModelNotFoundError(
                f"The model '{self.model}' is not installed in Ollama. Run `ollama pull {self.model}` and try again."
            )
        if response.status_code >= 400:
            detail = _error_detail(response)
            logger.error("Ollama %s %s failed with %s: %s", method, path, response.status_code, detail)
            raise OllamaError(f"Ollama returned an error ({response.status_code}): {detail}")
        return response


def _error_detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error") or response.text)
    except ValueError:
        return response.text or "unknown error"
