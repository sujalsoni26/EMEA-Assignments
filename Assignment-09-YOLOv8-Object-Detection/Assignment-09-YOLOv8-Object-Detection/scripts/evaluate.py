"""Evaluate the fine-tuned model (and the COCO baseline) on the held-out test split.

    python scripts/evaluate.py

Reports Precision, Recall, mAP@0.5 and mAP@0.5:0.95 overall and per class, and
compares the fine-tuned model with the original COCO-pretrained YOLOv8n on the
five classes both models know (Person, Mobile phone, Cup, Bottle, Laptop).
Writes ``reports/metrics.json`` and ``reports/evaluation.md`` and copies the most
useful plots into ``reports/``.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "oi-desk"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"

# fine-tuned class name -> COCO class name
SHARED = {"Person": "person", "Mobile phone": "cell phone", "Cup": "cup", "Bottle": "bottle", "Laptop": "laptop"}


def device() -> str:
    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def validate(weights: Path, data: Path, name: str) -> dict:
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data), split="test", imgsz=640, batch=16, device=device(), plots=True,
        project=str(RUNS / "eval"), name=name, exist_ok=True, verbose=False,
    )
    box = metrics.box
    per_class = {}
    for i, class_index in enumerate(box.ap_class_index):
        per_class[metrics.names[int(class_index)]] = {
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
            "mAP50": round(float(box.ap50[i]), 4),
            "mAP50-95": round(float(box.ap[i]), 4),
        }
    return {
        "overall": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "mAP50": round(float(box.map50), 4),
            "mAP50-95": round(float(box.map), 4),
        },
        "per_class": per_class,
        "speed_ms": {k: round(v, 2) for k, v in metrics.speed.items()},
        "save_dir": str(metrics.save_dir),
    }


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    finetuned_weights = MODELS / "yolov8n-oi-desk.pt"
    if not finetuned_weights.exists():
        sys.exit("models/yolov8n-oi-desk.pt not found. Run `python scripts/train.py` first.")
    REPORTS.mkdir(exist_ok=True)

    print("Evaluating the fine-tuned model on the test split…")
    finetuned = validate(finetuned_weights, DATASET / "data.yaml", "finetuned-test")
    print("Evaluating the COCO-pretrained baseline on the same images…")
    baseline = validate(MODELS / "yolov8n.pt", DATASET / "coco_eval" / "data.yaml", "coco-baseline-test")

    comparison = {}
    for ours, coco in SHARED.items():
        comparison[ours] = {
            "baseline": baseline["per_class"].get(coco, {}),
            "finetuned": finetuned["per_class"].get(ours, {}),
        }
    shared_summary = {
        model: {metric: mean([comparison[c][model][metric] for c in SHARED if comparison[c][model]])
                for metric in ("precision", "recall", "mAP50", "mAP50-95")}
        for model in ("baseline", "finetuned")
    }
    results = {"finetuned": finetuned, "baseline_coco": baseline, "shared_classes": comparison,
               "shared_summary": shared_summary}
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))

    stats = json.loads((REPORTS / "dataset_stats.json").read_text()) if (REPORTS / "dataset_stats.json").exists() else None
    lines = [
        "# Evaluation on the held-out test split",
        "",
        "Test images come from the official Open Images V7 **test** split and were never seen during training "
        "or model selection (the validation split was used for that).",
        "",
        "## Fine-tuned YOLOv8n — all classes",
        "",
        "| Class | Test images | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in finetuned["per_class"].items():
        images = stats["test"]["per_class"][name]["images"] if stats else "–"
        lines.append(f"| {name} | {images} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['mAP50'])} | {fmt(m['mAP50-95'])} |")
    o = finetuned["overall"]
    lines.append(f"| **All** | {stats['test']['images'] if stats else '–'} | **{fmt(o['precision'])}** | **{fmt(o['recall'])}** | "
                 f"**{fmt(o['mAP50'])}** | **{fmt(o['mAP50-95'])}** |")
    lines += [
        "",
        "## Before vs after fine-tuning (classes known to both models)",
        "",
        "The baseline is the official COCO-pretrained `yolov8n.pt`, evaluated on the same test images with labels "
        "mapped to COCO class ids. Glasses, Headphones and Watch are not COCO classes, so the baseline cannot detect "
        "them at all (recall 0).",
        "",
        "| Class | Baseline mAP@0.5 | Fine-tuned mAP@0.5 | Baseline mAP@0.5:0.95 | Fine-tuned mAP@0.5:0.95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, c in comparison.items():
        b, f = c["baseline"], c["finetuned"]
        if b and f:
            lines.append(f"| {name} | {fmt(b['mAP50'])} | {fmt(f['mAP50'])} | {fmt(b['mAP50-95'])} | {fmt(f['mAP50-95'])} |")
    sb, sf = shared_summary["baseline"], shared_summary["finetuned"]
    lines.append(f"| **Mean (5 classes)** | **{fmt(sb['mAP50'])}** | **{fmt(sf['mAP50'])}** | **{fmt(sb['mAP50-95'])}** | **{fmt(sf['mAP50-95'])}** |")
    lines += [
        "",
        f"Inference speed of the fine-tuned model on this machine ({device()}): "
        f"{finetuned['speed_ms'].get('inference', 0):.1f} ms per image at 640 px "
        f"(pre-process {finetuned['speed_ms'].get('preprocess', 0):.1f} ms, NMS {finetuned['speed_ms'].get('postprocess', 0):.1f} ms).",
        "",
    ]
    (REPORTS / "evaluation.md").write_text("\n".join(lines))

    # Copy the most useful plots next to the report.
    plots = REPORTS / "plots"
    plots.mkdir(exist_ok=True)
    eval_dir = Path(finetuned["save_dir"])
    for pattern in ("*PR_curve.png", "*F1_curve.png", "confusion_matrix_normalized.png", "val_batch*_pred.jpg", "val_batch*_labels.jpg"):
        for path in eval_dir.glob(pattern):
            shutil.copy2(path, plots / f"test_{path.name}")
    train_dir = RUNS / "train" / "oi-desk-yolov8n"
    for name in ("results.png", "results.csv", "labels.jpg"):
        if (train_dir / name).exists():
            shutil.copy2(train_dir / name, plots / f"train_{name}")

    print((REPORTS / "evaluation.md").read_text())
    print(f"Plots copied to {plots}")


if __name__ == "__main__":
    main()
