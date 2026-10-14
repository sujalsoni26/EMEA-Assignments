# Assignment 9 – Real-Time Object Detection using YOLOv8

A real-time object detection application built on the official **Ultralytics YOLOv8** implementation.

- A COCO-pretrained YOLOv8n is **fine-tuned on an Open Images V7 subset** of 8 everyday "desk & personal" classes.
- It is evaluated with **Precision, Recall and mAP** on the held-out Open Images test split and compared with the
  original pretrained model.
- It is served through a **FastAPI + HTML/CSS/JavaScript** web app that detects objects live from the browser
  webcam over a WebSocket.

![Live webcam detection](screenshots/Screenshot%202026-10-14%20214447.png)

## Requirements checklist

| Requirement | Where it is implemented |
|---|---|
| Load a suitable pretrained YOLOv8 model | `yolov8n.pt` (official COCO-pretrained Ultralytics checkpoint, chosen for real-time speed), loaded in [`scripts/train.py`](scripts/train.py) and [`app/detector.py`](app/detector.py) |
| Use a relevant dataset such as Open Images for fine-tuning or evaluation | [`scripts/prepare_dataset.py`](scripts/prepare_dataset.py) builds a YOLO-format subset of **Open Images V7** from the official train / validation / test splits ([dataset statistics](reports/dataset_stats.md)) |
| Train or fine-tune the model and review Precision, Recall and mAP | [`scripts/train.py`](scripts/train.py) (fine-tuning) and [`scripts/evaluate.py`](scripts/evaluate.py) (test-split metrics + baseline comparison). Results and review are [below](#results) and in [`reports/`](reports) |
| Perform inference and display bounding boxes, class labels and confidence scores | [`scripts/predict.py`](scripts/predict.py) (saves annotated images, see [`reports/predictions/`](reports/predictions)), plus the web app's canvas overlay and `POST /api/detect/annotated` |
| Integrate webcam-based detection using FastAPI, HTML, CSS and JavaScript | [`app/main.py`](app/main.py) (REST + WebSocket `/ws/detect`) and [`app/static/`](app/static) (`index.html`, `styles.css`, `app.js`) |

## Dataset: Open Images V7 subset

The full Open Images dataset is about 9M images, so [`scripts/prepare_dataset.py`](scripts/prepare_dataset.py)
builds a focused subset that is useful for a webcam:

| Class | Open Images labels merged into it | In COCO? |
|---|---|---|
| Person | Person, Man, Woman, Boy, Girl | yes (`person`) |
| Glasses | Glasses | **no** |
| Mobile phone | Mobile phone | yes (`cell phone`) |
| Cup | Coffee cup, Mug | yes (`cup`) |
| Bottle | Bottle | yes (`bottle`) |
| Laptop | Laptop | yes (`laptop`) |
| Headphones | Headphones | **no** |
| Watch | Watch | **no** |

How the subset is built:

- It uses the **official splits**: Open Images *train* for training, *validation* for model selection, and *test*
  only for the final metrics.
- Images containing *group-of* boxes (one box drawn around a crowd of objects) are skipped, because they make the
  labels ambiguous. So are images with an EXIF rotation.
- Sampling is balanced: about 600 training images per class, picking the rarest classes first.
- Images are resized so the longest side is 640 px. Labels are written in YOLO format (`class cx cy w h`, normalised).
- A COCO-labelled copy of the test split is also created, so the original pretrained model can be scored on exactly
  the same images.

The result is **4,011 train / 565 validation / 907 test images**; per-class counts are in
[`reports/dataset_stats.md`](reports/dataset_stats.md).

## Results

Fine-tuned for 15 epochs from `yolov8n.pt` (640 px, batch 16, Apple M4 GPU, about 55 min). Training curves:
[`reports/plots/train_results.png`](reports/plots/train_results.png).

### Precision, Recall and mAP on the held-out test split (907 images)

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Person | 0.917 | 0.519 | 0.688 | 0.472 |
| Glasses | 0.913 | 0.544 | 0.690 | 0.363 |
| Mobile phone | 0.717 | 0.862 | 0.859 | 0.748 |
| Cup | 0.871 | 0.739 | 0.780 | 0.663 |
| Bottle | 0.798 | 0.660 | 0.744 | 0.600 |
| Laptop | 0.837 | 0.758 | 0.834 | 0.674 |
| Headphones | 0.755 | 0.596 | 0.732 | 0.464 |
| Watch | 0.701 | 0.630 | 0.678 | 0.516 |
| **All classes** | **0.814** | **0.663** | **0.751** | **0.563** |

### Before vs after fine-tuning

The original COCO-pretrained `yolov8n.pt`, scored on the same test images for the 5 classes both models know:

| | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| Pretrained (COCO) | 0.702 | 0.570 |
| **Fine-tuned (Open Images)** | **0.781** | **0.632** |

### Review

- **Fine-tuning helped.** mAP@0.5 on the shared classes rose from 0.70 to 0.78, with the biggest gains on Cup
  (0.58 → 0.78) and Mobile phone (0.74 → 0.86).
- **New classes.** The model now also detects Glasses, Headphones and Watch, which the pretrained model can't detect
  at all.
- **Precision is higher than recall** (0.81 vs 0.66). The model is usually right when it detects something, but it
  misses some objects. Recall is lowest for Person (0.52): Open Images photos often contain small, distant or
  partly-visible people that a small model misses.
- **Small objects are harder.** Glasses and Watch have the lowest mAP@0.5:0.95, because they are small and need
  precise boxes.
- **Speed.** Inference takes about 2–4 ms per image on the Apple GPU, which is fast enough for real-time webcam use.

More detail: [`reports/evaluation.md`](reports/evaluation.md), PR curves and the confusion matrix in
[`reports/plots/`](reports/plots), and example predictions with boxes, labels and scores in
[`reports/predictions/`](reports/predictions).

## The web app

```
Browser                                              FastAPI (app/main.py)
getUserMedia ─► <video> ─► canvas (640 px) ─► JPEG ──WebSocket /ws/detect──► decode ─► YOLOv8 (MPS/CUDA/CPU)
<canvas> overlay ◄── boxes, labels, confidences ◄──────────── JSON ◄───────────────────┘
```

- **Live webcam tab.**
  - Start/stop the camera and pick a camera if there are several.
  - Switch between the fine-tuned model and the COCO baseline.
  - Adjust the confidence and IoU (NMS) thresholds live.
  - Show or hide labels, and save a snapshot (PNG with the boxes drawn).
  - See live FPS, inference time, round-trip time and the list of detections.
- **Upload tab.** Detect objects in an uploaded image.
- **Backpressure.** The browser sends the next frame only after the previous result arrives, so frames never pile up
  and latency stays low.
- **Robustness.**
  - Clear messages when camera permission is denied, no camera is found, or the camera is busy.
  - The WebSocket reconnects automatically.
  - The server rejects invalid images, out-of-range thresholds and unknown models with clear errors.
- **Model handling.** Inference runs in a thread pool with one lock per model (Ultralytics models aren't
  thread-safe), on the Apple GPU (MPS), CUDA or CPU, whichever is available.

| Endpoint | Description |
|---|---|
| `GET /` | Web page |
| `GET /api/models` | Available models and their classes |
| `POST /api/detect` | Image upload (`file`, optional `model`, `conf`, `iou`) → JSON boxes in pixel coordinates |
| `POST /api/detect/annotated` | Same, but returns a JPEG with boxes, labels and scores drawn |
| `WS /ws/detect` | Send `{"type":"config",...}` (optional) and binary JPEG frames; receive `{"type":"detections",...}` |

## How to run

```bash
# Option A: Using Conda
conda env create -f environment.yml
conda activate a09-yolov8

# Option B: Without Conda (Python 3.11 required)
# Run these commands from the project root directory
py -3.11 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 1. Build the dataset (downloads ~2.3 GB of annotation CSVs + ~5,500 images)
python scripts/prepare_dataset.py          # or --quick for a tiny smoke-test dataset

# 2. Fine-tune (Apple M4 GPU: about 4 min/epoch; also works on CUDA or CPU)
python scripts/train.py                    # -> models/yolov8n-oi-desk.pt, plots in runs/train/

# 3. Evaluate on the test split (+ COCO baseline comparison)
python scripts/evaluate.py                 # -> reports/evaluation.md, reports/metrics.json, reports/plots/

# 4. Inference on images / video / webcam (OpenCV window)
python scripts/predict.py                  # 8 random test images -> reports/predictions/
python scripts/predict.py --source my_photo.jpg
python scripts/predict.py --source 0       # webcam

# 5. Web app
python -m uvicorn app.main:app --port 8000 # open http://localhost:8000 and click "Start camera"
```

**Notes:**
- Install [Python 3.11](https://www.python.org/downloads/release/python-3119/) if it isn't already installed.
- Ensure `requirements.txt` is present in the project root.
- On Windows PowerShell, if activation is blocked by the execution policy, use Command Prompt and run `.venv\Scripts\activate.bat` instead.
- Activate the appropriate environment before running the dataset, training, evaluation, prediction, or web app commands.
- The `.venv` environment only needs to be created once; activate it again whenever you open a new terminal.


The fine-tuned weights are included in `models/`, so you can run step 5 straight away. Browsers only allow camera
access on `localhost` or over HTTPS.

## Tests

```bash
pytest     # 10 tests: REST + WebSocket API with a stub detector, error handling, and a real YOLOv8 inference check
```

## Project structure

```
app/                 FastAPI app (main.py), YOLO wrapper (detector.py), static/ (HTML, CSS, JS)
scripts/             prepare_dataset.py, train.py, evaluate.py, predict.py
models/              yolov8n.pt (COCO-pretrained), yolov8n-oi-desk.pt (fine-tuned, best checkpoint)
reports/             dataset_stats.md, evaluation.md, metrics.json, plots/, predictions/
screenshots/         screenshots of the web app
tests/
datasets/, runs/     generated by the scripts (not needed to run the app)
```
