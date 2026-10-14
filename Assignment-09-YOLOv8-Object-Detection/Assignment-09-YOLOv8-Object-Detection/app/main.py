"""FastAPI application: real-time YOLOv8 object detection from the browser webcam.

* ``GET  /``               web page (HTML, CSS, JavaScript)
* ``GET  /api/models``     available models and their classes
* ``POST /api/detect``     detect objects in an uploaded image (JSON)
* ``POST /api/detect/annotated``  same, but returns a JPEG with boxes drawn
* ``WS   /ws/detect``      real-time detection: the browser sends JPEG frames,
                           the server answers with boxes, labels and confidences
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .detector import MODEL_SPECS, DetectorRegistry, InvalidImageError, decode_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("yolo_app")

STATIC_DIR = Path(__file__).parent / "static"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_FRAME_BYTES = 2 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = DetectorRegistry()
    await run_in_threadpool(registry.load)
    if not registry.detectors:
        raise RuntimeError("No YOLO model could be loaded. See the log above.")
    app.state.registry = registry
    yield


app = FastAPI(title="YOLOv8 Real-Time Object Detection", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def check_thresholds(conf: float, iou: float) -> str | None:
    if not 0.01 <= conf <= 0.99:
        return "conf must be between 0.01 and 0.99."
    if not 0.05 <= iou <= 0.95:
        return "iou must be between 0.05 and 0.95."
    return None


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/models")
async def list_models(request: Request) -> dict:
    registry: DetectorRegistry = request.app.state.registry
    models = []
    for spec in MODEL_SPECS:
        detector = registry.detectors.get(spec.key)
        models.append({
            "key": spec.key,
            "label": spec.label,
            "description": spec.description,
            "available": detector is not None,
            "reason": registry.missing.get(spec.key),
            "classes": list(detector.names.values()) if detector else [],
        })
    return {"default": registry.default_key, "device": registry.device, "models": models}


async def _read_image(file: UploadFile) -> bytes:
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidImageError("The image is larger than 10 MB.")
    return data


@app.post("/api/detect")
async def detect(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(None),
    conf: float = Form(0.35),
    iou: float = Form(0.5),
):
    """Detect objects in an uploaded image and return boxes (pixel coordinates), labels and confidences."""
    if problem := check_thresholds(conf, iou):
        return error(422, "invalid_parameter", problem)
    try:
        detector = request.app.state.registry.get(model)
        image = decode_image(await _read_image(file))
    except KeyError as exc:
        return error(404, "model_not_available", str(exc.args[0]))
    except InvalidImageError as exc:
        return error(415, "invalid_image", str(exc))
    return await run_in_threadpool(detector.detect, image, conf, iou)


@app.post("/api/detect/annotated", responses={200: {"content": {"image/jpeg": {}}}})
async def detect_annotated(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(None),
    conf: float = Form(0.35),
    iou: float = Form(0.5),
):
    """Detect objects and return the image with bounding boxes, labels and confidence scores drawn on it."""
    if problem := check_thresholds(conf, iou):
        return error(422, "invalid_parameter", problem)
    try:
        detector = request.app.state.registry.get(model)
        image = decode_image(await _read_image(file))
    except KeyError as exc:
        return error(404, "model_not_available", str(exc.args[0]))
    except InvalidImageError as exc:
        return error(415, "invalid_image", str(exc))
    jpeg = await run_in_threadpool(detector.annotate, image, conf, iou)
    return Response(content=jpeg, media_type="image/jpeg")


@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket) -> None:
    """Real-time detection over a WebSocket.

    Client -> server: a JSON text message ``{"type": "config", "model": ..., "conf": ..., "iou": ...}``
    (optional, any time) and binary messages containing one JPEG frame each.
    Server -> client: ``{"type": "detections", ...}`` for every frame, or ``{"type": "error", "message": ...}``.
    The browser sends the next frame only after it receives the answer, so frames never pile up.
    """
    await websocket.accept()
    registry: DetectorRegistry = websocket.app.state.registry
    settings = {"model": registry.default_key, "conf": 0.35, "iou": 0.5}
    frames = 0
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("text") is not None:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") != "config":
                        raise ValueError("unknown message type")
                    conf = float(data.get("conf", settings["conf"]))
                    iou = float(data.get("iou", settings["iou"]))
                    if problem := check_thresholds(conf, iou):
                        raise ValueError(problem)
                    registry.get(data.get("model", settings["model"]))  # validates the model key
                    settings.update(model=data.get("model", settings["model"]), conf=conf, iou=iou)
                    await websocket.send_json({"type": "config", **settings})
                except (ValueError, KeyError, TypeError) as exc:
                    await websocket.send_json({"type": "error", "message": f"Invalid config: {exc}"})
                continue

            frame = message.get("bytes") or b""
            if len(frame) > MAX_FRAME_BYTES:
                await websocket.send_json({"type": "error", "message": "Frame too large (max 2 MB)."})
                continue
            try:
                image = decode_image(frame)
                detector = registry.get(settings["model"])
                result = await run_in_threadpool(detector.detect, image, settings["conf"], settings["iou"])
            except InvalidImageError as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
                continue
            frames += 1
            await websocket.send_json({"type": "detections", "frame": frames, **result})
    except WebSocketDisconnect:
        pass
    except (RuntimeError, asyncio.CancelledError):
        pass  # connection closed while sending
    finally:
        logger.info("WebSocket closed after %d frames", frames)
