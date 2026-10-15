# Architecture

## Overview

PageOCRDiff synthesises two research papers into a unified, reproducible page-level OCR pipeline:

| Component | Paper Source | Description |
|-----------|-------------|-------------|
| Four-head detector | Paper 1 (IEEE Access 2025) | Region, Affinity, Orientation (32-bin), Scale (10-bin) |
| Layout analysis | Paper 1 | Delaunay triangulation + polynomial line fitting |
| Reading order | Paper 1 | XY-cut columns + polynomial baseline sort |
| Word recognizer | Paper 1 | ResNet-CNN + CTC (optional BiLSTM) |
| Diffusion SR | Paper 2 (IEEE IoTJ 2024) | Conditional U-Net + Linear Attention + residual learning |

---

## Stage 1 — Four-Head Text Detector

```
Input image (B, 3, H, W)
       ↓
ResNet-18/34 backbone (ImageNet weights)
  stem → pool → layer1 → layer2 → layer3 → layer4
                 (64)     (128)    (256)    (512)
       ↓
U-Net decoder (skip connections)
  up4 → up3 → up2 → up1   [output stride = 2]
                           feature (B, 64, H/2, W/2)
       ↓  ↓  ↓  ↓
  head_region    → (B, 1, H/2, W/2)   sigmoid  [region heatmap]
  head_affinity  → (B, 1, H/2, W/2)   sigmoid  [affinity heatmap]
  head_orientation → (B, 32, H/2, W/2) logits  [32-class orientation]
  head_scale       → (B, 10, H/2, W/2) logits  [10-class scale]
```

### Loss (Paper 1 Eq. 1)
```
L = L_region + L_affinity + λ_s · L_scale + λ_o · L_orientation
```
- **L_region, L_affinity**: balanced MSE with OHEM (3:1 neg:pos ratio)
- **L_scale, L_orientation**: cross-entropy masked to text-positive pixels
- **λ_s = 0.1, λ_o = 0.3**

---

## Stage 2 — Word Recognizer (CTC)

```
Word crop (B, 1, 32, W)  grayscale, height-normalised
       ↓
CNN: 4 conv stages (32→64→128→256 channels)
     with max-pool, residual blocks, adaptive avg-pool → (B, 256, 1, W')
       ↓
Optional BiLSTM (2-layer, 256 hidden) → (B, W', 512)
       ↓
Linear classifier → (B, W', num_classes)
       ↓ transpose
(W', B, num_classes)  →  CTC loss
```

---

## Stage 3 — Layout Analysis Pipeline

### Connected Component Extraction
1. Combine `region_map` and `affinity_map` via element-wise max
2. Threshold → binary map → `cv2.connectedComponentsWithStats`
3. For each CC: compute text-pixel-weighted orientation and scale estimate

### Block Grouping — Delaunay Triangulation (Paper 1 Section III-D)
1. Build Delaunay triangulation over CC centers
2. Prune edge (p, q) when: `dist(p, q) ≥ ε · min(scale_p, scale_q)` where **ε = 2**
3. Connected subgraphs = text blocks

### Line Grouping — Polynomial Fitting (Paper 1 Section III-E)
1. For each block: sort CCs by rotated y-coordinate (angle from dominant orientation)
2. Grow lines by seeding unassigned CCs and adding candidates whose polynomial RMSE ≤ s̄/4
3. Fit polynomial of degree k ≤ 4 to final line members

### Reading Order (Paper 1 Section V)
1. **Inter-block**: XY-cut column detection → top-to-bottom within each column
2. **Inter-line**: sort by polynomial baseline f_i(x_avg)
3. **Intra-line**: left-to-right by rotated x-coordinate

---

## Stage 4 — OCR-Diff Super-Resolution (Paper 2)

```
LR crop (B, 3, h, w)
       ↓ bicubic upsample
x_up (B, 3, H, W)
       ↓
LR Feature Extractor: 5 residual blocks, skip connection
x_f (B, 3, H, W)

Diffusion sampling:
  X_T ∼ N(0, I)
  for t = T → 0:
    ε_θ = ConditionalUNet([X_t, x_f], t)   [6-channel input]
    X_{t-1} = DDIM_reverse(X_t, ε_θ)

Output: X̂_0 + x_up   [residual learning, Paper 2 Eq. 9]
```

### Conditional U-Net
- **Input**: concat(X_t, x_f) = 6 channels
- **Time embedding**: sinusoidal → MLP → injected via residual addition
- **Bottleneck**: Linear Attention O(N) — avoids quadratic softmax attention
- **Output**: 3-channel noise prediction

### Linear Attention (Paper 2 Section III-A)
```
Q = softmax(QK^T / √d)V   →   Ω(N²)  [standard, AVOIDED]
Linear: ctx = K^T V; out = Q · ctx    →   O(N)   [used]
```

---

## Post-processing

1. **CTC Greedy Decode**: collapse blanks and repeated tokens
2. **WordNinja**: segment run-on words into dictionary words
3. **Confidence Bleed-through Filter**: discard words with confidence < (μ − σ · bleed_sigma)
4. **Hyphenation merge**: re-join hyphen-broken line-end words

---

## Data Flow Summary

```
Camera image
  → Preprocess (CLAHE, deskew, pad to ÷32)
  → Detector → 4 maps
  → CC extraction (orientation, scale per CC)
  → Delaunay block grouping (ε=2)
  → Polynomial line grouping (deg≤4, RMSE≤s̄/4)
  → XY-cut reading order
  → [Optional] OCR-Diff SR on low-confidence crops
  → CTC Recognizer per word crop
  → WordNinja + confidence filter
  → Structured text output
```
