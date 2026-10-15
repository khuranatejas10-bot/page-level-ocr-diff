#!/usr/bin/env python
"""
scripts/finetune_diffusion_for_ocr.py
Stage 4b: Fine-tune OCR-Diff with recognition-aware loss.
Combines MSE noise prediction loss with CTC recognition loss on enhanced output.
Usage:
    python scripts/finetune_diffusion_for_ocr.py --config configs/smoke_test.yaml \
        --sr_ckpt runs/diffusion_sr/ckpt_pretrain_final.pt \
        --rec_ckpt runs/recognizer/ckpt_final.pt
"""
import argparse
import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from src.pageocrdiff.config import Config
from src.pageocrdiff.models import OCRDiffSR, CTCRecognizer, Vocabulary
from src.pageocrdiff.models.diffusion_sr import DiffusionSchedule
from src.pageocrdiff.data import SRPairDataset, RecognizerDataset, recognizer_collate
from src.pageocrdiff.data.synthetic_page_generator import SyntheticPageGenerator, generate_word_crops
from src.pageocrdiff.utils import set_seed, get_logger, AverageMeter, Checkpointer, save_metrics

logger = get_logger("finetune_diff_ocr")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune diffusion SR with recognition-aware loss")
    p.add_argument("--config", default="configs/smoke_test.yaml")
    p.add_argument("--sr_ckpt", default=None, help="Pretrained SR checkpoint")
    p.add_argument("--rec_ckpt", default=None, help="Recognizer checkpoint (frozen)")
    p.add_argument("--max_iter", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config) if os.path.exists(args.config) else Config()
    if args.max_iter:
        cfg.diffusion_sr.max_iterations = args.max_iter
    if args.output_dir:
        cfg.output_dir = args.output_dir

    device = cfg.resolve_device()
    set_seed(cfg.seed)

    # ── Load models ────────────────────────────────────────────────────────
    sr_model = OCRDiffSR(
        base_ch=cfg.diffusion_sr.base_channels,
        time_dim=cfg.diffusion_sr.time_emb_dim,
        T=cfg.diffusion_sr.T,
    ).to(device)
    if args.sr_ckpt and os.path.exists(args.sr_ckpt):
        Checkpointer.load(args.sr_ckpt, sr_model, device=device)
        logger.info(f"Loaded SR checkpoint: {args.sr_ckpt}")
    else:
        logger.warning("No SR checkpoint provided; training from scratch.")

    vocab = Vocabulary(cfg.recognizer.charset)
    rec_model = CTCRecognizer(num_classes=vocab.num_classes).to(device)
    if args.rec_ckpt and os.path.exists(args.rec_ckpt):
        Checkpointer.load(args.rec_ckpt, rec_model, device=device)
        logger.info(f"Loaded recognizer checkpoint: {args.rec_ckpt}")
    for p in rec_model.parameters():
        p.requires_grad_(False)  # Freeze recognizer
    rec_model.eval()

    # ── Data ───────────────────────────────────────────────────────────────
    import cv2
    data_root = cfg.data_root
    train_manifest = os.path.join(data_root, "train_manifest.json")
    word_crops = []
    if os.path.exists(train_manifest):
        with open(train_manifest) as f:
            manifest = json.load(f)
        for s in manifest.get("samples", [])[:50]:  # Limit for fine-tuning
            img_path = os.path.join(data_root, s["image"])
            anno_path = os.path.join(data_root, s["annotation"])
            img = cv2.imread(img_path)
            if img is None or not os.path.exists(anno_path):
                continue
            with open(anno_path) as f:
                anno = json.load(f)
            for wd in anno.get("words", []):
                x, y, w, h = int(wd["x"]), int(wd["y"]), int(wd["w"]), int(wd["h"])
                ih, iw = img.shape[:2]
                crop = img[max(0, y-2):min(ih, y+h+2), max(0, x-2):min(iw, x+w+2)]
                if crop.shape[0] > 4 and crop.shape[1] > 4:
                    word_crops.append((crop, wd.get("text", "")))
    else:
        gen = SyntheticPageGenerator(seed=cfg.seed, img_size=256)
        for _ in range(30):
            page = gen.generate_page()
            word_crops.extend(generate_word_crops(page["words"], page["image"]))

    sr_ds = SRPairDataset(word_crops,
                           hr_size=(cfg.diffusion_sr.hr_height, cfg.diffusion_sr.hr_width),
                           lr_size=(cfg.diffusion_sr.lr_height, cfg.diffusion_sr.lr_width))
    loader = DataLoader(sr_ds, batch_size=min(8, cfg.diffusion_sr.batch_size),
                        shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    schedule = DiffusionSchedule(T=cfg.diffusion_sr.T, device=device)
    optimizer = torch.optim.Adam(sr_model.parameters(), lr=cfg.diffusion_sr.lr_finetune)
    ctc_loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    run_dir = os.path.join(cfg.output_dir, "diffusion_sr")
    ckptr = Checkpointer(run_dir)
    metrics_path = os.path.join(run_dir, "finetune_metrics.jsonl")

    max_iter = max(100, cfg.diffusion_sr.max_iterations // 5)  # Shorter fine-tuning
    logger.info(f"Fine-tuning diffusion SR: {len(sr_ds)} pairs, max_iter={max_iter}")

    sr_model.train()
    mse_meter = AverageMeter("mse")
    total_meter = AverageMeter("total")
    data_iter = iter(loader)
    step = 0
    t0 = time.time()

    while step < max_iter:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        hr = batch["hr"].to(device)
        lr_up = batch["lr_up"].to(device)
        B = hr.shape[0]

        t_step = torch.randint(0, cfg.diffusion_sr.T, (B,), device=device, dtype=torch.long)
        x_t, noise = schedule.q_sample(hr, t_step)

        optimizer.zero_grad()
        pred_noise = sr_model(x_t, lr_up, t_step)
        mse = F.mse_loss(pred_noise, noise)

        # Recognition-aware: reconstruct x0, run recognizer, compute CTC
        ab = schedule.alpha_bar[t_step].reshape(-1, 1, 1, 1)
        x0_pred = (x_t - torch.sqrt(1 - ab) * pred_noise) / torch.sqrt(ab.clamp(min=1e-8))
        x0_pred = x0_pred.clamp(-1, 1)

        # Convert to grayscale for recognizer
        gray = (0.299 * x0_pred[:, 0] + 0.587 * x0_pred[:, 1] + 0.114 * x0_pred[:, 2])
        gray = gray.unsqueeze(1)  # (B, 1, H, W)
        gray = F.interpolate(gray, size=(cfg.recognizer.img_height, cfg.recognizer.img_width),
                             mode="bilinear", align_corners=False)

        logits = rec_model(gray)  # (T, B, C)
        T_len = logits.shape[0]
        input_lens = torch.full((B,), T_len, dtype=torch.long)

        # Dummy labels (self-supervised: use greedy decode as pseudo-target)
        with torch.no_grad():
            preds = logits.argmax(dim=2)  # (T, B)
        labels_list = []
        label_lens = []
        for b in range(B):
            seq = preds[:, b].tolist()
            decoded = []
            prev = -1
            for idx in seq:
                if idx != 0 and idx != prev:
                    decoded.append(idx)
                prev = idx
            decoded = decoded or [1]
            labels_list.extend(decoded)
            label_lens.append(len(decoded))

        labels_t = torch.tensor(labels_list, dtype=torch.long)
        label_lens_t = torch.tensor(label_lens, dtype=torch.long)

        rec_loss = ctc_loss_fn(logits.log_softmax(2).cpu(), labels_t, input_lens, label_lens_t)
        if not torch.isfinite(rec_loss):
            rec_loss = torch.tensor(0.0)

        total = cfg.diffusion_sr.recon_weight * mse + cfg.diffusion_sr.recog_weight * rec_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(sr_model.parameters(), 5.0)
        optimizer.step()

        mse_meter.update(mse.item(), B)
        total_meter.update(total.item(), B)
        step += 1

        if step % 50 == 0:
            logger.info(f"[{step:>5}/{max_iter}] mse={mse_meter.avg:.6f} total={total_meter.avg:.6f} "
                        f"elapsed={time.time()-t0:.0f}s")
            save_metrics({"step": step, "mse": mse_meter.avg, "total": total_meter.avg}, metrics_path)
            mse_meter.reset(); total_meter.reset()

        if step % 200 == 0 or step == max_iter:
            ckptr.save(f"ft_step{step}", sr_model, optimizer, step)

    logger.info("Diffusion SR fine-tuning complete.")
    ckptr.save("finetune_final", sr_model, optimizer, step)


if __name__ == "__main__":
    main()
