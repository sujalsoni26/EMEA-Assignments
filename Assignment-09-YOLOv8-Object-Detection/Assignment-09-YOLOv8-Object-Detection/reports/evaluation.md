# Evaluation on the held-out test split

Test images come from the official Open Images V7 **test** split and were never seen during training or model selection (the validation split was used for that).

## Fine-tuned YOLOv8n — all classes

| Class | Test images | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| Person | 293 | 0.917 | 0.519 | 0.688 | 0.472 |
| Glasses | 150 | 0.913 | 0.544 | 0.690 | 0.363 |
| Mobile phone | 150 | 0.717 | 0.862 | 0.859 | 0.748 |
| Cup | 150 | 0.871 | 0.739 | 0.780 | 0.663 |
| Bottle | 151 | 0.798 | 0.660 | 0.744 | 0.600 |
| Laptop | 150 | 0.837 | 0.758 | 0.834 | 0.674 |
| Headphones | 94 | 0.755 | 0.596 | 0.732 | 0.464 |
| Watch | 88 | 0.701 | 0.630 | 0.678 | 0.516 |
| **All** | 907 | **0.814** | **0.663** | **0.751** | **0.563** |

## Before vs after fine-tuning (classes known to both models)

The baseline is the official COCO-pretrained `yolov8n.pt`, evaluated on the same test images with labels mapped to COCO class ids. Glasses, Headphones and Watch are not COCO classes, so the baseline cannot detect them at all (recall 0).

| Class | Baseline mAP@0.5 | Fine-tuned mAP@0.5 | Baseline mAP@0.5:0.95 | Fine-tuned mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Person | 0.712 | 0.688 | 0.535 | 0.472 |
| Mobile phone | 0.743 | 0.859 | 0.636 | 0.748 |
| Cup | 0.579 | 0.780 | 0.474 | 0.663 |
| Bottle | 0.696 | 0.744 | 0.555 | 0.600 |
| Laptop | 0.776 | 0.834 | 0.652 | 0.674 |
| **Mean (5 classes)** | **0.702** | **0.781** | **0.570** | **0.632** |

Inference speed of the fine-tuned model on this machine (mps): 1.7 ms per image at 640 px (pre-process 0.1 ms, NMS 4.9 ms).
