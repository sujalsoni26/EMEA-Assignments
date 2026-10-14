"""API and WebSocket tests. A stub detector replaces YOLO so the tests are fast;
one optional test runs the real COCO model when models/yolov8n.pt is present."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.detector import MODEL_SPECS, InvalidImageError, decode_image


def jpeg(width: int = 320, height: int = 240) -> bytes:
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (200, 180), (0, 0, 255), -1)
    return cv2.imencode(".jpg", image)[1].tobytes()


class StubDetector:
    def __init__(self, key: str):
        self.spec = next(s for s in MODEL_SPECS if s.key == key)
        self.names = {0: "Person", 1: "Cup"}
        self.calls: list[tuple[float, float]] = []

    def detect(self, image, conf=0.35, iou=0.5, imgsz=640):
        self.calls.append((conf, iou))
        h, w = image.shape[:2]
        return {"model": self.spec.key, "width": w, "height": h, "inference_ms": 5.0, "total_ms": 6.0,
                "detections": [{"class_id": 1, "label": "Cup", "confidence": 0.91, "box": [40.0, 40.0, 200.0, 180.0]}]}

    def annotate(self, image, conf=0.35, iou=0.5):
        return cv2.imencode(".jpg", image)[1].tobytes()


class StubRegistry:
    def __init__(self, *args, **kwargs):
        self.device = "cpu"
        self.detectors = {"oi-desk": StubDetector("oi-desk"), "coco": StubDetector("coco")}
        self.missing = {}

    def load(self):
        return None

    @property
    def default_key(self):
        return "oi-desk"

    def get(self, key):
        key = key or self.default_key
        if key not in self.detectors:
            raise KeyError(f"Model '{key}' is not available: unknown model")
        return self.detectors[key]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "DetectorRegistry", StubRegistry)
    with TestClient(main.app) as test_client:
        yield test_client


def test_index_and_models(client):
    assert "YOLOv8 Real-Time Object Detection" in client.get("/").text
    data = client.get("/api/models").json()
    assert data["default"] == "oi-desk"
    assert {m["key"] for m in data["models"]} == {"oi-desk", "coco"}


def test_detect_image(client):
    response = client.post("/api/detect", files={"file": ("x.jpg", jpeg(), "image/jpeg")}, data={"conf": "0.5", "model": "coco"})
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "coco"
    assert (data["width"], data["height"]) == (320, 240)
    assert data["detections"][0]["label"] == "Cup"


def test_detect_annotated_returns_jpeg(client):
    response = client.post("/api/detect/annotated", files={"file": ("x.jpg", jpeg(), "image/jpeg")})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"


@pytest.mark.parametrize(
    "files, data, status, code",
    [
        ({"file": ("x.txt", b"not an image", "text/plain")}, {}, 415, "invalid_image"),
        ({"file": ("x.jpg", jpeg(), "image/jpeg")}, {"conf": "1.5"}, 422, "invalid_parameter"),
        ({"file": ("x.jpg", jpeg(), "image/jpeg")}, {"model": "yolo-x"}, 404, "model_not_available"),
    ],
)
def test_detect_errors(client, files, data, status, code):
    response = client.post("/api/detect", files=files, data=data)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_websocket_stream(client):
    with client.websocket_connect("/ws/detect") as ws:
        ws.send_json({"type": "config", "model": "coco", "conf": 0.6, "iou": 0.4})
        assert ws.receive_json() == {"type": "config", "model": "coco", "conf": 0.6, "iou": 0.4}
        for frame_number in (1, 2):
            ws.send_bytes(jpeg())
            message = ws.receive_json()
            assert message["type"] == "detections"
            assert message["frame"] == frame_number
            assert message["model"] == "coco"
            assert message["detections"][0]["confidence"] == 0.91


def test_websocket_errors(client):
    with client.websocket_connect("/ws/detect") as ws:
        ws.send_json({"type": "config", "model": "nope"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "config", "conf": 5})
        assert "conf" in ws.receive_json()["message"]
        ws.send_bytes(b"garbage")
        assert ws.receive_json() == {"type": "error", "message": "The file is not a valid image (supported: JPEG, PNG, WebP, BMP)."}


def test_decode_image_rejects_bad_input():
    with pytest.raises(InvalidImageError):
        decode_image(b"")
    with pytest.raises(InvalidImageError):
        decode_image(b"not an image")
    assert decode_image(jpeg()).shape == (240, 320, 3)


@pytest.mark.skipif(not (Path(__file__).parents[1] / "models" / "yolov8n.pt").exists(), reason="COCO weights not downloaded")
def test_real_coco_model_detects_a_person():
    import ultralytics
    from app.detector import Detector

    bus = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    detector = Detector(next(s for s in MODEL_SPECS if s.key == "coco"), "cpu")
    result = detector.detect(cv2.imread(str(bus)))
    labels = {d["label"] for d in result["detections"]}
    assert {"person", "bus"} <= labels
    assert all(0 <= d["confidence"] <= 1 for d in result["detections"])
