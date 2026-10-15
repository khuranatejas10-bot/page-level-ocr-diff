# Training Guide

## Prerequisites

```bash
pip install -r requirements.txt
```

## Step 0 — Generate Synthetic Data

```bash
# Smoke test (fast)
python scripts/generate_synthetic_data.py \
    --output data/synthetic --train 200 --val 40 --test 40 --img_size 768

# Large-scale training
python scripts/generate_synthetic_data.py \
    --output data/synthetic --train 5000 --val 500 --test 500
```

## Step 0b — Inspect Dataset

```bash
python scripts/inspect_dataset.py \
    --manifest data/synthetic/train_manifest.json \
    --data_root data/synthetic \
    --visualise --n_samples 8 --out_dir reports/inspect
```

## Step 1 — Train Detector (Stage 1)

```bash
python scripts/train_detector.py --config configs/default.yaml

# Resume from checkpoint:
python scripts/train_detector.py --config configs/default.yaml \
    --resume runs/detector/ckpt_step5000.pt
```

**Key hyperparameters (configs/default.yaml):**
| Param | Value | Paper reference |
|-------|-------|----------------|
| backbone | resnet18 | Paper 1 |
| image_size | 768×768 | Paper 1 |
| batch_size | 4 | - |
| lr | 1e-4 → decay ×0.8 / 2000 iter | Paper 1 |
| max_iterations | 50,000 | Paper 1 |
| λ_scale | 0.1 | Paper 1 |
| λ_orientation | 0.3 | Paper 1 |
| num_orientation_bins | 32 | Paper 1 |
| num_scale_bins | 10 | Paper 1 |

## Step 2 — Train Recognizer (Stage 2)

```bash
python scripts/train_recognizer.py --config configs/default.yaml
```

**Key hyperparameters:**
| Param | Value |
|-------|-------|
| loss | CTC (blank=0) |
| img_height | 32 |
| img_width | 128 (max, padded) |
| lr | 1e-4 |
| batch_size | 64 |
| max_iterations | 30,000 |
| grad_clip | 5.0 |

## Step 3 — Pretrain Diffusion SR (Stage 4a)

```bash
python scripts/train_diffusion_sr.py --config configs/default.yaml
```

**Trains OCR-Diff with MSE noise prediction (Paper 2 Stage 1):**
- Forward diffusion: `x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε`
- Loss: `L_simple = ‖ε_θ(x_t, x_f, t) − ε‖²`
- Schedule: cosine (Nichol & Dhariwal), T=1000

## Step 4 — Fine-tune with Recognition Loss (Stage 4b)

```bash
python scripts/finetune_diffusion_for_ocr.py --config configs/default.yaml \
    --sr_ckpt runs/diffusion_sr/ckpt_pretrain_final.pt \
    --rec_ckpt runs/recognizer/ckpt_final.pt
```

**Combined loss:**
```
L_total = L_recon + λ_rec · L_ctc
λ_rec = 0.01
```

## Smoke Test

Run the complete smoke test (verifies all stages in ~2 minutes on CPU):

```bash
python scripts/run_smoke_test.py --quick
```

Expected output: 7 PASS steps and a report at `reports/smoke_test_report.json`.

## Running Unit Tests

```bash
python -m pytest tests/ -q
```

## Monitoring

Training metrics are saved as JSONL files:
- `runs/detector/train_metrics.jsonl`
- `runs/recognizer/train_metrics.jsonl`
- `runs/diffusion_sr/pretrain_metrics.jsonl`
- `runs/diffusion_sr/finetune_metrics.jsonl`

Parse with:
```python
import json
with open("runs/detector/train_metrics.jsonl") as f:
    records = [json.loads(l) for l in f]
```
