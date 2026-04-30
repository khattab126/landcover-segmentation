# Land Cover Segmentation with Partial Focal Cross-Entropy Loss

An interactive web application for satellite image land cover segmentation using a U-Net trained on sparse point labels with a partial focal cross-entropy (pfCE) loss function.

**Live Demo:** [Streamlit App](https://khattab126-landcover-segmentation-app.streamlit.app)

---

## Overview

This project addresses **weakly-supervised semantic segmentation** — training a segmentation model using only a handful of labeled points per image instead of dense pixel-wise annotations. This dramatically reduces labeling cost while achieving competitive results.

### Key idea

Instead of labeling every pixel in a 2448×2448 satellite image (~6 million pixels), an annotator clicks just 20–50 points per class. The model learns from these sparse labels using a **partial focal cross-entropy loss** that:

1. **Masks the loss** to labeled points only (`MASK_labeled`)
2. **Down-weights easy pixels** using focal modulation `(1 - p_t)^γ`

The loss formula:

```
pfCE = Σ [ FocalLoss(pred, GT) · MASK_labeled ] / Σ MASK_labeled
```

---

## Dataset

**DeepGlobe Land Cover Classification** (2018 challenge) — 300 satellite tiles at 50 cm/pixel resolution, 2448×2448 pixels.

| Class | Color | Description |
|-------|-------|-------------|
| Urban | Cyan | Buildings, roads, infrastructure |
| Agriculture | Yellow | Farmland, crops |
| Rangeland | Magenta | Grass, shrubs, open areas |
| Forest | Green | Trees, dense vegetation |
| Water | Blue | Rivers, lakes, ponds |
| Barren | White | Bare soil, rock, sand |

---

## Model Architecture

A compact **U-Net** with 4-level encoder/decoder (~7M parameters):
- Encoder: 4 `DoubleConv` blocks with `MaxPool2d` downsampling
- Decoder: `ConvTranspose2d` upsampling with skip connections
- Output: 1×1 convolution → 6-class logits
- Training resolution: 256×256

---

## Loss Function: `PartialFocalCELoss`

A drop-in replacement for `nn.CrossEntropyLoss`:

- Computes per-pixel `log_softmax` and extracts `p_t` via `gather`
- Applies focal modulation: `(1 - p_t)^γ · (-log p_t)`
- Multiplies by binary `MASK_labeled` (1 at labeled points, 0 elsewhere)
- Averages over the number of labeled pixels

With `γ = 0`, the loss reduces exactly to `F.cross_entropy(..., ignore_index=255)`.

---

## Experiments & Results

### Experiment 1 — Annotation Budget

How many labeled points per class are needed?

| Points/class | Best mIoU | Best Epoch | Pixel Accuracy |
|---:|---:|---:|---:|
| 1 | 0.249 | 12 | 0.574 |
| 5 | 0.254 | 10 | 0.591 |
| 10 | 0.271 | 6 | 0.574 |
| 20 | 0.276 | 10 | 0.619 |
| 50 | **0.302** | 11 | 0.540 |
| Full supervision | 0.229 | 10 | 0.629 |

**Takeaway:** mIoU rises monotonically but flattens above ~20 points/class — diminishing returns.

### Experiment 2 — Loss × Sampling Strategy

At a fixed budget of 100 points/image:

| Configuration | Best mIoU | Pixel Accuracy |
|---|---:|---:|
| pCE + Balanced | 0.299 | 0.624 |
| pfCE + Balanced | 0.276 | 0.619 |
| pCE + Uniform | 0.250 | 0.603 |
| **pfCE + Uniform** | **0.304** | **0.643** |

**Takeaway:** Balanced sampling and focal modulation are substitutes, not complements. In the realistic case (uniform clicks), focal CE is the best default.

---

## Web App Features

The deployed Streamlit app has three tabs:

| Tab | Description |
|-----|-------------|
| **Predict** | Select any of the 300 dataset images (or upload your own) and see real-time land cover predictions |
| **Experiments** | Interactive charts and tables for all experiment results, training curves, and per-class IoU comparisons |
| **Gallery** | Browse all 300 samples with satellite image, ground truth mask, and model prediction side-by-side |

---

## Tech Stack

- **Model:** PyTorch — custom U-Net + partial focal CE loss
- **Dataset:** DeepGlobe Land Cover via Kaggle (`kagglehub`)
- **Web App:** Streamlit + Plotly interactive charts
- **Deployment:** Streamlit Cloud (CPU inference)
- **Notebook:** Jupyter — full training pipeline with experiments and analysis

---

## Project Structure

```
├── app.py                          # Streamlit web app (3 tabs)
├── model.py                        # SmallUNet, inference, and utilities
├── model/
│   └── best_model.pth              # Saved model weights (7.4 MB)
├── results/
│   └── experiment_data.py          # Hardcoded experiment results
├── full_dataset/
│   ├── images/                     # 300 satellite images (256×256)
│   ├── masks/                      # 300 ground truth masks
│   └── predictions/                # 300 model predictions
├── assessment.ipynb                # Full training notebook
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/khattab126/landcover-segmentation.git
cd landcover-segmentation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Author

**Youssef Ahmed Saad** — Technical assessment for land cover segmentation with partial supervision.
