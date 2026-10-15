# Evaluation Guide

## Run Evaluation

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --det_ckpt runs/detector/ckpt_final.pt \
    --rec_ckpt runs/recognizer/ckpt_final.pt \
    --manifest data/synthetic/test_manifest.json \
    --output reports/evaluation_results.json
```

## Metrics Reference

### Text Quality Metrics

| Metric | Formula | Range | Paper |
|--------|---------|-------|-------|
| **CER** | `edit(hyp, ref) / len(ref)` | ≥0 (lower=better) | Paper 1 |
| **WER** | `word_edit(hyp, ref) / len(ref_words)` | ≥0 (lower=better) | Paper 1 |
| **Edit Distance** | Levenshtein(hyp, ref) | ≥0 (lower=better) | Both |

### Layout / Detection Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **IoU** | Box intersection-over-union | [0, 1] (higher=better) |
| **Precision** | TP / (TP+FP) at IoU≥0.5 | [0, 1] |
| **Recall** | TP / (TP+FN) at IoU≥0.5 | [0, 1] |
| **F1** | 2·P·R / (P+R) | [0, 1] |

### Block Order Metric

| Metric | Description | Range |
|--------|-------------|-------|
| **TBR-L** | Text-Block ROUGE-L using LCS of block sequence | [0, 1] (higher=better) |

**TBR-L Formula:**
```
P_lcs = LCS(pred_blocks, ref_blocks) / |pred_blocks|
R_lcs = LCS(pred_blocks, ref_blocks) / |ref_blocks|
TBR-L = 2·P_lcs·R_lcs / (P_lcs + R_lcs)
```

### SR Quality Metrics

| Metric | Formula | Range |
|--------|---------|-------|
| **PSNR** | 10·log₁₀(MAX²/MSE) | dB (higher=better) |
| **SSIM** | Structural similarity | [0, 1] (higher=better) |

## Interpreting Results

Example output (`reports/evaluation_results.json`):
```json
{
  "summary": {
    "cer": 0.12,
    "wer": 0.18,
    "tbrl": 0.85
  },
  "per_page": [
    {"id": "test_000000", "cer": 0.10, "wer": 0.15, "tbrl": 0.90, ...}
  ]
}
```

**Baseline expectations on synthetic data (untrained models):**
- CER: ~0.95 (random predictions)
- WER: ~1.0
- TBR-L: ~0.1–0.3 (random block order)

**After smoke-test training (5–10 iterations):**
- Models will not converge but the pipeline should run without errors.

**After full training (50K detector + 30K recognizer iterations):**
- CER: 0.05–0.15 (depends on data complexity)
- WER: 0.08–0.20
- TBR-L: 0.75–0.95

## Per-Component Evaluation

### Detection-only (word boxes)
```bash
python scripts/evaluate.py \
    --manifest data/synthetic/test_manifest.json \
    --iou_threshold 0.5
```
Outputs: `word_precision`, `word_recall`, `word_f1`, `mean_iou`.

### OCR-Diff SR Quality
After training the SR model, evaluate PSNR/SSIM:
```python
from src.pageocrdiff.metrics import psnr, ssim
# Compare enhanced crops to HR ground truth
p = psnr(hr_img / 255.0, enhanced / 255.0)
s = ssim(hr_img / 255.0, enhanced / 255.0)
```
