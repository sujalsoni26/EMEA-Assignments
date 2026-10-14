"""Run inference and save images with bounding boxes, class labels and confidence scores.

    python scripts/predict.py --source path/to/image.jpg
    python scripts/predict.py --source path/to/folder --model models/yolov8n.pt
    python scripts/predict.py --source 0            # webcam (OpenCV window, press q to quit)
    python scripts/predict.py                       # 8 random test images -> reports/predictions/
"""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def print_detections(result) -> None:
    name = Path(result.path).name
    if result.boxes is None or not len(result.boxes):
        print(f"{name}: no objects")
        return
    print(f"{name}:")
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = result.names[int(box.cls)]
        print(f"  {label:<14} {float(box.conf):.2f}  box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")


def run_webcam(model: YOLO, index: int, conf: float) -> None:
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise SystemExit(f"Could not open webcam {index}")
    print("Press q to quit")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.predict(frame, conf=conf, verbose=False)[0]
        cv2.imshow("YOLOv8 detection", result.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=None, help="image, folder, video or webcam index")
    parser.add_argument("--model", default=str(ROOT / "models" / "yolov8n-oi-desk.pt"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--output", default=None, help="output folder (default: runs/predict or reports/predictions)")
    parser.add_argument("--samples", type=int, default=8, help="number of random test images when --source is omitted")
    args = parser.parse_args()

    model = YOLO(args.model)
    if args.source is not None and args.source.isdigit():
        run_webcam(model, int(args.source), args.conf)
        return

    if args.source is None:
        images = sorted((ROOT / "datasets" / "oi-desk" / "images" / "test").glob("*.jpg"))
        random.Random(7).shuffle(images)
        source_dir = Path(tempfile.mkdtemp())
        for image in images[: args.samples]:
            shutil.copy2(image, source_dir / image.name)
        source = str(source_dir)
        output = Path(args.output or ROOT / "reports" / "predictions")
    else:
        source = args.source
        output = Path(args.output or ROOT / "runs" / "predict")

    output.mkdir(parents=True, exist_ok=True)
    for result in model.predict(source, conf=args.conf, stream=True, verbose=False):
        print_detections(result)
        cv2.imwrite(str(output / f"{Path(result.path).stem}_pred.jpg"), result.plot(line_width=2))
    print(f"\nAnnotated images saved to {output}")


if __name__ == "__main__":
    main()
