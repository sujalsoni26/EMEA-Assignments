"""FastAPI application: Local LLM Text Summarizer (TinyLlama via Ollama)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .ollama_client import OllamaClient, OllamaError
from .schemas import (
    ConfigResponse,
    ErrorResponse,
    HealthResponse,
    SummarizeRequest,
    SummarizeResponse,
    count_words,
)
from .summarizer import Summarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("summarizer")

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ollama = OllamaClient(settings.ollama_base_url, settings.ollama_model, settings.request_timeout)
    logger.info("Using Ollama at %s with model '%s'", settings.ollama_base_url, settings.ollama_model)
    yield
    await app.state.ollama.aclose()


app = FastAPI(
    title="Local LLM Text Summarizer",
    description="Summarize text with TinyLlama running locally through Ollama.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_ollama(request: Request) -> OllamaClient:
    return request.app.state.ollama


def get_summarizer(client: OllamaClient = Depends(get_ollama)) -> Summarizer:
    return Summarizer(client, settings.chunk_chars)


# --------------------------------------------------------------------------- #
# Error handling: every error is returned as {"error": {"code", "message"}}
# --------------------------------------------------------------------------- #
def error_response(status_code: int, code: str, message: str, details: list | None = None) -> JSONResponse:
    body: dict = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=jsonable_encoder(body))


def _readable_validation_message(error: dict) -> str:
    location = [str(part) for part in error.get("loc", []) if part != "body"]
    message = str(error.get("msg", "Invalid input."))
    if message.startswith("Value error, "):
        return message[len("Value error, "):]
    if error.get("type") == "json_invalid":
        return "The request body is not valid JSON."
    if error.get("type") == "missing":
        return f"The field '{'.'.join(location) or 'body'}' is required."
    if error.get("type") == "enum" and location:
        return f"Invalid value for '{location[-1]}': {message}."
    if location:
        return f"Invalid value for '{'.'.join(location)}': {message}."
    return message


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    message = _readable_validation_message(errors[0]) if errors else "Invalid request."
    details = [{"field": ".".join(str(p) for p in e.get("loc", [])), "message": _readable_validation_message(e)} for e in errors]
    return error_response(422, "validation_error", message, details)


@app.exception_handler(OllamaError)
async def ollama_exception_handler(request: Request, exc: OllamaError) -> JSONResponse:
    logger.warning("Ollama error (%s): %s", exc.code, exc.message)
    return error_response(exc.status_code, exc.code, exc.message)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unexpected error while handling %s %s", request.method, request.url.path)
    return error_response(500, "internal_error", "An unexpected error occurred. Please try again.")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config", response_model=ConfigResponse, tags=["meta"])
async def get_config() -> ConfigResponse:
    """Limits used for input validation (the web page uses them for client-side checks)."""
    return ConfigResponse(model=settings.ollama_model, min_words=settings.min_words, max_chars=settings.max_chars)


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
async def health(client: OllamaClient = Depends(get_ollama)) -> HealthResponse:
    """Check that Ollama is reachable and the TinyLlama model is installed."""
    try:
        available = await client.is_model_available()
    except OllamaError as exc:
        return HealthResponse(
            status="unavailable", ollama_reachable=False, model=client.model, model_available=False, message=exc.message
        )
    if not available:
        return HealthResponse(
            status="degraded",
            ollama_reachable=True,
            model=client.model,
            model_available=False,
            message=f"Ollama is running but the model '{client.model}' is missing. Run `ollama pull {client.model}`.",
        )
    return HealthResponse(
        status="ok", ollama_reachable=True, model=client.model, model_available=True, message="Ready to summarize."
    )


@app.post(
    "/api/summarize",
    response_model=SummarizeResponse,
    tags=["summarization"],
    responses={
        422: {"model": ErrorResponse, "description": "Invalid input (empty, too short or too long text)."},
        502: {"model": ErrorResponse, "description": "The model returned an invalid or empty response."},
        503: {"model": ErrorResponse, "description": "Ollama is not running or the model is not installed."},
        504: {"model": ErrorResponse, "description": "The model did not respond in time."},
    },
)
async def summarize(payload: SummarizeRequest, summarizer: Summarizer = Depends(get_summarizer)) -> SummarizeResponse:
    """Generate a concise summary of the submitted text."""
    result = await summarizer.summarize(payload.text, payload.length, payload.format)
    return SummarizeResponse(
        summary=result.summary,
        format=payload.format,
        length=payload.length,
        model=summarizer.client.model,
        input_words=count_words(payload.text),
        summary_words=count_words(" ".join(line.lstrip("- ") for line in result.summary.splitlines())),
        chunks=result.chunks,
        duration_ms=result.duration_ms,
    )
