"""Build a YOLO-format subset of Open Images V7 for fine-tuning YOLOv8.

    python scripts/prepare_dataset.py            # full dataset (default quotas)
    python scripts/prepare_dataset.py --quick    # tiny dataset for a smoke test

What it does
------------
1. Downloads the official Open Images bounding-box annotation CSVs (train,
   validation and test splits) and the class-description file.
2. Keeps the boxes of the 8 target classes defined in ``CLASSES`` (some Open
   Images labels are merged, e.g. Man/Woman/Boy/Girl/Person -> Person).
3. Drops images that contain "group-of" boxes for these classes (a single box
   around a crowd of objects), because they make the labels ambiguous.
4. Samples a balanced set of images per class and split (rarest classes first).
5. Downloads the images from the public Open Images S3 bucket, skips images with
   an EXIF rotation, resizes them so the longest side is 640 px, and writes the
   YOLO label files (class cx cy w h, normalised to 0-1).
6. Writes ``data.yaml`` for Ultralytics, a COCO-labelled copy of the test split
   (``coco_eval/``) used to evaluate the *pre-trained* COCO model as a baseline,
   and dataset statistics in ``reports/dataset_stats.md``.

Open Images is released under CC BY 4.0 (annotations) and the images are listed
as having a CC BY 2.0 licence.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets" / "open-images" / "annotations"
OUT_DIR = ROOT / "datasets" / "oi-desk"
REPORTS_DIR = ROOT / "reports"

ANNOTATION_URLS = {
    "class-descriptions": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"

# Target class -> Open Images display names merged into it.
CLASSES: dict[str, list[str]] = {
    "Person": ["Person", "Man", "Woman", "Boy", "Girl"],
    "Glasses": ["Glasses"],
    "Mobile phone": ["Mobile phone"],
    "Cup": ["Coffee cup", "Mug"],
    "Bottle": ["Bottle"],
    "Laptop": ["Laptop"],
    "Headphones": ["Headphones"],
    "Watch": ["Watch"],
}
CLASS_NAMES = list(CLASSES)

# Our classes that also exist in COCO (used to evaluate the pre-trained COCO model).
COCO_IDS = {"Person": 0, "Bottle": 39, "Cup": 41, "Laptop": 63, "Mobile phone": 67}

# Output split name -> (Open Images split, images per class)
SPLITS = {"train": ("train", 600), "val": ("validation", 120), "test": ("test", 150)}
QUICK_SPLITS = {"train": ("validation", 15), "val": ("validation", 5), "test": ("test", 5)}

IMG_SIZE = 640
SEED = 0


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #
def download_file(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for block in response.iter_content(chunk_size=1 << 20):
                fh.write(block)
                bar.update(len(block))
    tmp.replace(dest)
    return dest


def load_label_ids() -> dict[str, int]:
    """Map Open Images label ids (e.g. /m/01g317) to our class index."""
    path = download_file(ANNOTATION_URLS["class-descriptions"], RAW_DIR / "oidv7-class-descriptions-boxable.csv")
    desc = pd.read_csv(path)
    name_to_id = dict(zip(desc["DisplayName"], desc["LabelName"]))
    mapping: dict[str, int] = {}
    for class_index, (class_name, oi_names) in enumerate(CLASSES.items()):
        for oi_name in oi_names:
            if oi_name not in name_to_id:
                sys.exit(f"Open Images class '{oi_name}' not found in the class descriptions file")
            mapping[name_to_id[oi_name]] = class_index
    return mapping


def load_boxes(oi_split: str, label_ids: dict[str, int]) -> pd.DataFrame:
    filename = Path(ANNOTATION_URLS[oi_split]).name
    path = download_file(ANNOTATION_URLS[oi_split], RAW_DIR / filename)
    columns = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf"]
    parts = []
    # The train CSV is 2.2 GB, so read it in chunks and keep only our classes.
    for chunk in tqdm(pd.read_csv(path, usecols=columns, chunksize=2_000_000), desc=f"Filtering {filename}"):
        parts.append(chunk[chunk["LabelName"].isin(label_ids)])
    df = pd.concat(parts, ignore_index=True)
    df["cls"] = df["LabelName"].map(label_ids).astype(int)
    group_images = set(df.loc[df["IsGroupOf"] == 1, "ImageID"])
    return df[~df["ImageID"].isin(group_images)].reset_index(drop=True)


def sample_images(df: pd.DataFrame, per_class: int, rng: np.random.Generator, exclude: set[str]) -> list[str]:
    """Pick up to ``per_class`` images for each class, rarest classes first."""
    images_per_class = {c: set(df.loc[df["cls"] == c, "ImageID"]) - exclude for c in range(len(CLASS_NAMES))}
    selected: set[str] = set()
    for c in sorted(images_per_class, key=lambda k: len(images_per_class[k])):
        have = len(images_per_class[c] & selected)
        need = per_class - have
        if need <= 0:
            continue
        candidates = sorted(images_per_class[c] - selected)
        picked = rng.choice(candidates, size=min(need, len(candidates)), replace=False)
        selected.update(picked.tolist())
    return sorted(selected)


# --------------------------------------------------------------------------- #
# Images and labels
# --------------------------------------------------------------------------- #
def fetch_image(session: requests.Session, oi_split: str, image_id: str, dest: Path) -> str:
    """Download, validate and resize one image. Returns 'ok', 'cached', 'rotated' or 'failed'."""
    if dest.exists():
        return "cached"
    url = IMAGE_URL.format(split=oi_split, image_id=image_id)
    for _ in range(3):
        try:
            response = session.get(url, timeout=30)
            if response.status_code != 200:
                continue
            image = Image.open(io.BytesIO(response.content))
            # Images with an EXIF rotation are skipped: it is ambiguous whether
            # the boxes refer to the stored or the rotated orientation.
            if image.getexif().get(0x0112, 1) != 1:
                return "rotated"
            image = image.convert("RGB")
            image.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            tmp = dest.with_suffix(".part")
            image.save(tmp, format="JPEG", quality=90)
            tmp.replace(dest)
            return "ok"
        except (requests.RequestException, OSError):
            continue
    return "failed"


def write_label(path: Path, boxes: pd.DataFrame, class_map: dict[int, int] | None = None) -> int:
    lines = []
    for row in boxes.itertuples(index=False):
        cls = row.cls if class_map is None else class_map.get(row.cls)
        if cls is None:
            continue
        x1, x2 = np.clip([row.XMin, row.XMax], 0, 1)
        y1, y2 = np.clip([row.YMin, row.YMax], 0, 1)
        w, h = x2 - x1, y2 - y1
        if w <= 0.002 or h <= 0.002:  # degenerate box
            continue
        lines.append(f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def build_split(name: str, oi_split: str, image_ids: list[str], boxes: pd.DataFrame, workers: int) -> dict:
    img_dir = OUT_DIR / "images" / name
    lbl_dir = OUT_DIR / "labels" / name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    session.mount("https://", adapter)
    status: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_image, session, oi_split, i, img_dir / f"{i}.jpg"): i for i in image_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {name} images"):
            status[futures[future]] = future.result()

    kept = [i for i in image_ids if status[i] in ("ok", "cached")]
    # Remove files left over from earlier runs, which would otherwise be used as background images.
    keep_set = set(kept)
    for stale in [p for p in img_dir.glob("*.jpg") if p.stem not in keep_set] + list(lbl_dir.glob("*.txt")):
        stale.unlink()
    by_image = boxes[boxes["ImageID"].isin(set(kept))].groupby("ImageID")
    class_images = [0] * len(CLASS_NAMES)
    class_boxes = [0] * len(CLASS_NAMES)
    for image_id, image_boxes in by_image:
        write_label(lbl_dir / f"{image_id}.txt", image_boxes)
        for c, n in image_boxes["cls"].value_counts().items():
            class_images[c] += 1
            class_boxes[c] += int(n)
    counts = {s: sum(1 for v in status.values() if v == s) for s in ("ok", "cached", "rotated", "failed")}
    print(f"{name}: kept {len(kept)} images {counts}")
    return {
        "images": len(kept),
        "download": counts,
        "per_class": {CLASS_NAMES[c]: {"images": class_images[c], "boxes": class_boxes[c]} for c in range(len(CLASS_NAMES))},
    }


def build_coco_eval(boxes: pd.DataFrame) -> None:
    """Copy of the test split whose labels use COCO class ids (only the 5 shared classes),
    so the pre-trained COCO model can be scored on exactly the same images."""
    coco_dir = OUT_DIR / "coco_eval"
    if coco_dir.exists():
        shutil.rmtree(coco_dir)
    (coco_dir / "images" / "test").mkdir(parents=True)
    (coco_dir / "labels" / "test").mkdir(parents=True)
    class_map = {CLASS_NAMES.index(name): coco_id for name, coco_id in COCO_IDS.items()}
    for image in sorted((OUT_DIR / "images" / "test").glob("*.jpg")):
        os.link(image, coco_dir / "images" / "test" / image.name)  # hard link: no extra disk space
        write_label(coco_dir / "labels" / "test" / f"{image.stem}.txt", boxes[boxes["ImageID"] == image.stem], class_map)

    import ultralytics
    import yaml

    coco_yaml = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml"
    coco_names = yaml.safe_load(coco_yaml.read_text())["names"]
    lines = ["# Test split relabelled with COCO class ids (generated by prepare_dataset.py)", "train: images/test  # unused, required by Ultralytics", "test: images/test", "val: images/test", "names:"]
    lines += [f"  {i}: {coco_names[i]}" for i in range(80)]
    (coco_dir / "data.yaml").write_text("\n".join(lines) + "\n")


def write_data_yaml() -> None:
    lines = [
        "# Open Images V7 subset for YOLOv8 fine-tuning (generated by prepare_dataset.py)",
        "# Paths are relative to this file.",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    (OUT_DIR / "data.yaml").write_text("\n".join(lines) + "\n")


def write_stats(stats: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    lines = [
        "# Dataset statistics (Open Images V7 subset)",
        "",
        "| Class | Merged Open Images labels | Train images | Train boxes | Val images | Val boxes | Test images | Test boxes |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, oi_names in CLASSES.items():
        row = [name, ", ".join(oi_names)]
        for split in ("train", "val", "test"):
            c = stats[split]["per_class"][name]
            row += [str(c["images"]), str(c["boxes"])]
        lines.append("| " + " | ".join(row) + " |")
    lines.append(
        "| **Total images** | | **{}** | | **{}** | | **{}** | |".format(
            stats["train"]["images"], stats["val"]["images"], stats["test"]["images"]
        )
    )
    (REPORTS_DIR / "dataset_stats.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="build a tiny dataset (no 2.2 GB train CSV download)")
    parser.add_argument("--workers", type=int, default=32, help="parallel image downloads")
    args = parser.parse_args()

    splits = QUICK_SPLITS if args.quick else SPLITS
    rng = np.random.default_rng(SEED)
    label_ids = load_label_ids()
    boxes_cache: dict[str, pd.DataFrame] = {}
    used: set[str] = set()
    stats: dict = {"classes": CLASSES}

    for name, (oi_split, per_class) in splits.items():
        if oi_split not in boxes_cache:
            boxes_cache[oi_split] = load_boxes(oi_split, label_ids)
        boxes = boxes_cache[oi_split]
        image_ids = sample_images(boxes, per_class, rng, exclude=used)
        used.update(image_ids)  # in --quick mode train and val come from the same OI split
        print(f"{name}: sampled {len(image_ids)} images from Open Images '{oi_split}'")
        stats[name] = build_split(name, oi_split, image_ids, boxes, args.workers)

    write_data_yaml()
    build_coco_eval(boxes_cache[splits["test"][0]])
    write_stats(stats)
    print(f"\nDataset ready: {OUT_DIR / 'data.yaml'}")
    print(f"Statistics:    {REPORTS_DIR / 'dataset_stats.md'}")


if __name__ == "__main__":
    main()
