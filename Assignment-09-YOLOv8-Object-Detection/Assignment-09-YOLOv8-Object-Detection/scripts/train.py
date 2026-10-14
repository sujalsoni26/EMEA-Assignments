"""Fine-tune a COCO-pretrained YOLOv8 model on the Open Images subset.

    python scripts/train.py                       # defaults: yolov8n, 15 epochs, 640 px
    python scripts/train.py --epochs 5 --name smoke-test

Training curves, PR/F1 curves, confusion matrices and validation predictions are
written to ``runs/train/<name>/``. The best checkpoint (highest validation
fitness = 0.1*mAP50 + 0.9*mAP50-95) is copied to ``models/yolov8n-oi-desk.pt``,
which the web app uses.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "oi-desk" / "data.yaml"
MODELS_DIR = ROOT / "models"


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"  # Apple-silicon GPU
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolov8n.pt", help="pre-trained checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10, help="early-stopping patience (epochs)")
    parser.add_argument("--device", default=None, help="cpu, mps or a CUDA device id (default: auto)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="oi-desk-yolov8n")
    parser.add_argument("--output", default=str(MODELS_DIR / "yolov8n-oi-desk.pt"), help="where to copy best.pt")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        sys.exit(f"{DATA_YAML} not found. Run `python scripts/prepare_dataset.py` first.")

    MODELS_DIR.mkdir(exist_ok=True)
    weights = MODELS_DIR / args.model  # pre-trained weights are downloaded here on first use
    model = YOLO(str(weights))
    device = pick_device(args.device)
    print(f"Fine-tuning {args.model} on {DATA_YAML} using device '{device}'")

    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        workers=args.workers,
        project=str(ROOT / "runs" / "train"),
        name=args.name,
        exist_ok=True,
        seed=0,
        deterministic=False,  # deterministic mode is much slower on Apple GPUs
        plots=True,
        close_mosaic=5,  # disable mosaic augmentation for the last 5 epochs
    )

    best = Path(model.trainer.best)
    if not best.exists():
        sys.exit("Training finished but no best.pt was produced.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output)
    print(f"\nBest checkpoint copied to {output}")
    print(f"Training plots: {best.parents[1]}")


if __name__ == "__main__":
    main()
