# Soybean Disease Classification — Two-Stage Deep Learning Pipeline

A two-stage deep learning system for automated soybean disease detection, developed as part of an internship at IIIT Bangalore. The pipeline addresses a fundamental weakness of single-stage classifiers — fragility to out-of-distribution inputs — by sequentially gating images through a leaf detector before reaching the disease classifier.

**Authors:** SVN Sai Sathvik, Kotyada Parthiv  
**Institution:** International Institute of Information Technology Bangalore  
**Supervisor:** Sourab

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Disease Classes](#disease-classes)
- [Results](#results)
- [Datasets](#datasets)
- [Repository Structure](#repository-structure)
- [Setup & Usage](#setup--usage)
- [Model Weights](#model-weights)
- [Key Design Decisions](#key-design-decisions)

---

## Overview

Single-stage disease classifiers fail in field conditions because they assign a disease label to every input — including soil, hands, stems, and unrelated objects — with no way to reject non-leaf images. This project decomposes the problem into two sequential stages:

- **Stage 1 — Leaf Detection:** A **ResNet18** binary classifier accepts only images containing a leaf and rejects everything else.
- **Stage 2 — Disease Classification:** A MobileNetV2 multi-class classifier runs only on confirmed leaf images, assigning one of four disease/health labels.

This design eliminates spurious predictions, reduces wasted inference, and produces a system lightweight enough for edge and mobile deployment.

---

## Pipeline Architecture

```
Input Image
     │
     ▼
┌─────────────────────────────┐
│  Stage 1: ResNet            │
│  Leaf / Non-Leaf Classifier │
└─────────────────────────────┘
     │               │
   Leaf           Non-Leaf
     │               │
     ▼               ▼
┌──────────────┐   REJECT
│  Stage 2:    │   (pipeline halts)
│  MobileNetV2 │
│  Disease     │
│  Classifier  │
└──────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│  Healthy │ Yellow Mosaic │ Rust │ SDS    │
└──────────────────────────────────────────┘
```

---

## Disease Classes

| Class ID | Class Name | Description |
|---|---|---|
| 0 | Healthy | Normal leaf tissue; no visible disease symptoms |
| 1 | Yellow Mosaic | Viral-like mosaic patches and chlorosis across the leaf blade |
| 2 | Rust | Fungal pustules and bronzing on leaf underside and margins |
| 3 | Sudden Death Syndrome (SDS) | Necrotic streaking and defoliation symptoms |

---

## Results

### Stage 1 — Leaf Detector (ResNet)

| Metric | Value |
|---|---|
| Architecture | ResNet (ImageNet pre-trained) |
| Task | Binary classification (leaf / non-leaf) |
| Validation Accuracy | **99.07%** |
| Comparison (YOLO) | ~94% |

ResNet18 was selected over YOLO because the task requires only presence/absence detection (no bounding box needed), where a classification model is both faster and more accurate.

### Stage 2 — Disease Classifier (MobileNetV2)

| Metric | Value |
|---|---|
| Architecture | MobileNetV2 (ImageNet pre-trained) |
| Test Accuracy | **86.32%** |
| Validation Accuracy | **87.10%** |
| Macro Precision | 0.8815 |
| Macro Recall | 0.8632 |
| Macro F1 Score | 0.8565 |
| Validation Loss | 0.1173 |

**Per-class breakdown:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Healthy | 0.97 | 1.00 | 0.98 | 29 |
| Yellow Mosaic | 0.92 | 0.55 | 0.69 | 22 |
| Rust | 0.70 | 0.95 | 0.81 | 22 |
| SDS | 0.91 | 0.91 | 0.91 | 22 |
| **Macro avg** | **0.87** | **0.85** | **0.85** | 95 |

Yellow Mosaic shows the weakest recall (0.55), likely due to visual overlap with Rust (shared chlorotic patterns). This is the primary target for future improvement.

---

## Datasets

Both datasets are sourced from Kaggle:

| Stage | Dataset | Classes | Notes |
|---|---|---|---|
| Stage 1 | [Leaf vs. Non-Leaf Images](https://www.kaggle.com/datasets/robiulhasanjisan/leaf-vs-non-leaf-images/data) | 2 (leaf, non-leaf) | Diverse backgrounds, lighting, and angles |
| Stage 2 | [Soybean Leaf Image Dataset](https://www.kaggle.com/datasets/mamun009/soybean-leaf-image-dataset) | 4 disease/health classes | Only 4 target classes retained |

**Preprocessing (both stages):**
- Resize to 224×224 pixels
- ImageNet normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`)
- Augmentation: random horizontal flip + random rotation
- Stratified train/val/test split

---

## Repository Structure

```
Soyabean-Disease-Classification/
├── internship-different.ipynb                          # Main training and evaluation notebook
├── inference.py                                        # Standalone two-stage inference script
├── requirements.txt                                    # Pinned Python dependencies
├── best_mobilenet_soybean_model.pth                    # Stage 2 base checkpoint
├── best_mobilenet_soybean_model_finetuned.pth          # Stage 2 fine-tuned checkpoint
├── best_mobilenet_soybean_model_trained_all_layers.pth # Stage 2 all-layers checkpoint
├── Internship_report (1).pdf                           # Full internship report
└── Progress-Update-...pptx                             # Mid-internship progress slides
```

---

## Setup & Usage

**Requirements:**
```bash
pip install -r requirements.txt
```

**Run the notebook:**
```bash
jupyter notebook internship-different.ipynb
```

The notebook covers end-to-end: dataset loading, preprocessing, Stage 1 training, Stage 2 training, evaluation, and qualitative prediction visualization.

**Standalone inference (no Jupyter needed):**
```bash
python inference.py \
    --image path/to/your/leaf.jpg \
    --stage1_weights leaf_classifier_resnet18.pth \
    --stage2_weights best_mobilenet_soybean_model_finetuned.pth
```

---

## Model Weights

Three Stage 2 checkpoints are included, representing different training strategies:

| File | Description |
|---|---|
| `best_mobilenet_soybean_model.pth` | Classifier head only trained (backbone frozen) |
| `best_mobilenet_soybean_model_finetuned.pth` | End-to-end fine-tuning from frozen checkpoint |
| `best_mobilenet_soybean_model_trained_all_layers.pth` | All layers trained with ImageNet init |

To load a checkpoint:
```python
import torch
from torchvision import models

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(1280, 4)
model.load_state_dict(torch.load("best_mobilenet_soybean_model_finetuned.pth"))
model.eval()
```

---

## Key Design Decisions

**Why two stages instead of one?** Single-stage classifiers assign disease labels to all inputs regardless of content. In field conditions, cameras capture soil, hands, stems, and equipment — all of which a single classifier will confidently misclassify as diseased. The Stage 1 gate eliminates this entire failure mode.

**Why ResNet18 for Stage 1?** ResNet18 outperforms YOLO (99.07% vs ~94%) for the leaf/non-leaf binary task because bounding box localization is unnecessary — presence/absence detection is a simpler problem that a lightweight classifier handles better with less compute.

**Why MobileNetV2 for Stage 2?** MobileNetV2's depthwise separable convolutions keep parameter count low while preserving discriminative texture and color features critical for disease differentiation. It is directly exportable to TFLite/ONNX for edge deployment on mobile or embedded agricultural devices.

**Why only 10 epochs?** Training was deliberately kept conservative. The close alignment between validation accuracy (87.10%) and test accuracy (86.32%) confirms no overfitting — the model has learned generalizable disease patterns rather than dataset-specific artifacts.

---

## Known Limitations & Future Work

- **Yellow Mosaic recall is 0.55** — the weakest class due to visual similarity with Rust. Planned fix: class-weighted `CrossEntropyLoss` or Focal Loss.
- **Stage 1 weights not yet uploaded** — `leaf_classifier_resnet18.pth` needs to be added for end-to-end reproducibility.
- **No ONNX/TFLite export** — model export for edge deployment is a planned next step.
