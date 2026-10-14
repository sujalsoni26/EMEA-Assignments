"""YOLOv8 model loading and inference."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    path: Path
    label: str
    description: str


MODEL_SPECS = [
    ModelSpec(
        "oi-desk",
        MODELS_DIR / "yolov8n-oi-desk.pt",
        "YOLOv8n fine-tuned on Open Images (8 classes)",
        "Person, Glasses, Mobile phone, Cup, Bottle, Laptop, Headphones, Watch",
    ),
    ModelSpec(
        "coco",
        MODELS_DIR / "yolov8n.pt",
        "YOLOv8n pre-trained on COCO (80 classes)",
        "Official Ultralytics checkpoint, before fine-tuning",
    ),
]


class InvalidImageError(ValueError):
    pass


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise InvalidImageError("The image is empty.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("The file is not a valid image (supported: JPEG, PNG, WebP, BMP).")
    return image


class Detector:
    """One YOLO model. Ultralytics models are not thread-safe, so inference is serialised with a lock."""

    def __init__(self, spec: ModelSpec, device: str):
        self.spec = spec
        self.device = device
        self.model = YOLO(str(spec.path))
        self.names: dict[int, str] = self.model.names
        self._lock = threading.Lock()
        # Warm-up so the first real frame is not slow.
        self.model.predict(np.zeros((480, 640, 3), dtype=np.uint8), device=device, verbose=False)

    def detect(self, image: np.ndarray, conf: float = 0.35, iou: float = 0.5, imgsz: int = 640) -> dict:
        started = time.perf_counter()
        with self._lock:
            result = self.model.predict(image, conf=conf, iou=iou, imgsz=imgsz, device=self.device, verbose=False)[0]
        boxes = result.boxes
        detections = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, cls in zip(xyxy, confs, classes):
                detections.append({
                    "class_id": int(cls),
                    "label": self.names[int(cls)],
                    "confidence": round(float(score), 4),
                    "box": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
                })
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        height, width = image.shape[:2]
        return {
            "model": self.spec.key,
            "width": width,
            "height": height,
            "detections": detections,
            "inference_ms": round(result.speed.get("inference", 0.0), 1),
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    def annotate(self, image: np.ndarray, conf: float = 0.35, iou: float = 0.5) -> bytes:
        """Return a JPEG with boxes, class labels and confidence scores drawn by Ultralytics."""
        with self._lock:
            result = self.model.predict(image, conf=conf, iou=iou, device=self.device, verbose=False)[0]
        ok, buffer = cv2.imencode(".jpg", result.plot(line_width=2), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Could not encode the annotated image.")
        return buffer.tobytes()


class DetectorRegistry:
    def __init__(self, device: str | None = None):
        self.device = device or pick_device()
        self.detectors: dict[str, Detector] = {}
        self.missing: dict[str, str] = {}

    def load(self) -> None:
        for spec in MODEL_SPECS:
            if spec.key == "coco" and not spec.path.exists():
                # The official checkpoint is downloaded automatically by Ultralytics.
                spec.path.parent.mkdir(parents=True, exist_ok=True)
            if spec.key != "coco" and not spec.path.exists():
                self.missing[spec.key] = f"{spec.path.name} not found. Train it with `python scripts/train.py`."
                logger.warning("Model %s not available: %s", spec.key, self.missing[spec.key])
                continue
            try:
                self.detectors[spec.key] = Detector(spec, self.device)
                logger.info("Loaded %s from %s on %s", spec.key, spec.path, self.device)
            except Exception as exc:  # pragma: no cover - depends on local files / network
                self.missing[spec.key] = f"Could not load {spec.path.name}: {exc}"
                logger.exception("Could not load model %s", spec.key)

    @property
    def default_key(self) -> str | None:
        return next((spec.key for spec in MODEL_SPECS if spec.key in self.detectors), None)

    def get(self, key: str | None) -> Detector:
        key = key or self.default_key
        if key not in self.detectors:
            reason = self.missing.get(key or "", "unknown model")
            raise KeyError(f"Model '{key}' is not available: {reason}")
        return self.detectors[key]
