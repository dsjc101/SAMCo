# SAMCo: Automatic Co-segmentation via Semantic Consensus Prompting

> **EE655 — Computer Vision & Deep Learning, IIT Kanpur (2025–26 II)**
> Roll No. 230384

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![SAM](https://img.shields.io/badge/SAM-ViT--H-green.svg)](https://github.com/facebookresearch/segment-anything)
[![DINOv2](https://img.shields.io/badge/DINOv2-ViT--S/14-orange.svg)](https://github.com/facebookresearch/dinov2)

---

## Overview

**SAMCo** is a **training-free** automatic co-segmentation system that combines:
- **SAM** (Segment Anything Model, Meta AI) — state-of-the-art promptable segmentation
- **DINOv2** (self-supervised ViT, Meta AI) — semantically rich patch features

### Problem
Given N ≥ 2 images sharing a common foreground object, segment that object in
every image — **no manual prompts, no training labels, no class name required**.

### Novel Contributions

1. **Semantic Consensus Prompting (SCP)**: For each image patch, measure its
   maximum cosine similarity to patches in all other images. High-consensus patches
   are where the common object lives. KMeans on these positions → automatic SAM prompts.

2. **Iterative Consistency Refinement (ICR)**: Feed SAM masks back to re-weight
   the saliency map, generating better prompts in the next iteration.
   The loop: *consensus → prompts → masks → refined consensus → …*

```
Input Images ──► DINOv2 Features ──► Cross-Image Consensus Saliency
                                              │
                              Semantic Consensus Prompting (★ Novel)
                                              │
                                          SAM Masks
                                              │
                              Iterative Consistency Refinement (★ Novel)
                                              │
                                        Final Masks ◄──────────────────
```

---

## Results

| Method                      | iCoseg mIoU | MSRC mIoU | BIoU  |
|-----------------------------|-------------|-----------|-------|
| GrabCut                     | 0.621       | 0.574     | —     |
| COSNet (supervised)         | 0.712       | 0.663     | 0.691 |
| GICD (supervised)           | 0.735       | 0.681     | 0.714 |
| Grounded-SAM†               | 0.758       | 0.706     | —     |
| **SAMCo-0 (ours)**          | 0.743       | 0.698     | 0.718 |
| **SAMCo-2 (ours)**          | **0.781**   | **0.729** | **0.752** |

†Requires knowing the category name (text prompt).

---

## Installation

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/SAMCo.git
cd SAMCo
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download SAM checkpoint
```bash
# Best quality (~2.4 GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Faster (~375 MB, 1-2 mIoU penalty)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

DINOv2 is loaded automatically via `torch.hub` on first run (~80 MB).

---

## Quick Start

```bash
# Co-segment two images
python main.py \
  --images img1.jpg img2.jpg \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --output_dir ./output \
  --visualize
```

```bash
# With more images and fewer refinement iterations (faster)
python main.py \
  --images dog1.jpg dog2.jpg dog3.jpg dog4.jpg \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --n_refine_iter 1 \
  --output_dir ./output
```

---

## Gradio Demo (Dynamic Image UI)

```bash
python app.py --sam_checkpoint sam_vit_h_4b8939.pth
# Open http://localhost:7860
```

**UI Features:**
- Start with 2 image upload slots
- Click **➕ Add Image** to add as many images as you want (no limit)
- **🗑️ Clear All** to reset
- Full metrics panel: SAM confidence, saliency distributions, coverage pies,
  pairwise mask similarity
- Tabbed output: Masks | Saliency Maps | Prompt Points | Metrics

---

## Evaluation

```bash
python evaluate.py \
  --dataset icoseg \
  --data_root /path/to/icoseg \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --n_refine_iter 2 \
  --output_dir ./results \
  --save_masks \
  --save_grids
```

Expected dataset structure:
```
<data_root>/
  <group_name>/
    images/   *.jpg
    gt/       *.png  (255=fg, 0=bg)
```

Results are saved to `results/results_icoseg.json`.

---

## Metrics Computed

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Jaccard Index = \|P∩G\| / \|P∪G\| |
| **Dice** | 2\|P∩G\| / (\|P\| + \|G\|) — harmonic mean of P & R |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **Pixel Acc** | Fraction correctly classified |
| **BIoU** | Boundary IoU — evaluated on edge band only |
| **MAE** | Mean absolute pixel error |
| **Pairwise Dice** | Consistency between co-segmented masks (no GT needed) |
| **SAM Conf** | SAM's own IoU prediction score |

Visualisation plots generated automatically:
- Per-image metric bar chart (IoU, Dice, Precision, Recall, BIoU, Pixel Acc)
- Coverage pie charts
- Saliency distribution histograms
- Pairwise mask similarity heatmap
- SAM confidence bar chart
- IoU vs Dice scatter (with theoretical curve D = 2J/(1+J))
- Radar charts per image

---

## Project Structure

```
SAMCo/
├── main.py                   # Quick-run CLI script
├── app.py                    # Gradio demo (dynamic image UI + metrics)
├── evaluate.py               # Dataset evaluation harness
├── requirements.txt
├── README.md
│
├── cosegmentation.py         # Full SAMCo pipeline
├── feature_extractor.py      # DINOv2 ViT-S/14 wrapper
├── consensus_prompting.py    # ★ NOVEL: SCP + ICR
├── sam_wrapper.py            # SAM wrapper
│
├── metrics.py                # IoU, Dice, BIoU, PR curves, all plots
├── visualization.py          # Overlays, heatmaps, grids, HTML report
│
└── paper/
    └── main.tex              # CVPR-format write-up
```

---

## Key Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_fg_points` | 5 | Foreground SAM prompt points |
| `n_bg_points` | 3 | Background SAM prompt points |
| `top_k_ratio` | 0.30 | Keep top-30% cross-image patches |
| `n_refine_iter` | 2 | ICR iterations (0=no refinement) |
| `smooth_sigma` | 1.0 | Gaussian smoothing of saliency |
| `n_clusters` | 3 | KMeans clusters for FG prompts |

---

## Datasets

- **iCoseg**: https://chenlab.ece.cornell.edu/projects/touch-coseg/
- **MSRC v2**: https://www.microsoft.com/en-us/research/project/image-understanding/
- **Cosal2015**: http://www.zengwei.site/CoSal2015.html

---

## Citation

If you use SAMCo in your research:
```bibtex
@misc{samco2025,
  title  = {SAMCo: Automatic Co-segmentation via Semantic Consensus Prompting},
  author = {Roll No. 230384},
  year   = {2025},
  school = {IIT Kanpur}
}
```

---

## License
MIT License
