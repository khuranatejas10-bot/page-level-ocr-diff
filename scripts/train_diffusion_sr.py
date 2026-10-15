#!/usr/bin/env python
"""
scripts/train_diffusion_sr.py
Stage 4a: Pretrain OCR-Diff with MSE noise prediction on LR/HR pairs.
Usage:
    python scripts/train_diffusion_sr.py --config configs/smoke_test.yaml
"""
import argparse
import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from src.pageocrdiff.config import Config
from src.pageocrdiff.models import OCRDiffSR
from src.pageocrdiff.models.diffusion_sr import DiffusionSchedule
from src.pageocrdiff.data import SRPairDataset
from src.pageocrdiff.data.synthetic_page_generator import SyntheticPageGenerator, generate_word_crops
from src.pageocrdiff.utils import set_seed, get_logger, AverageMeter, Checkpointer, save_metrics

logger = get_logger("train_diffusion_sr")


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain OCR-Diff SR model")
    p.add_argument("--config", default="configs/smoke_test.yaml")
    p.add_argument("--max_iter", type=int, default=None)
    p.add_argument("--resume", default=None)
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
    logger.info(f"Device: {device}")

    # ── Build SR training pairs ────────────────────────────────────────────
    data_root = cfg.data_root
    train_manifest = os.path.join(data_root, "train_manifest.json")
    import cv2
    word_crops = []
    if os.path.exists(train_manifest):
        with open(train_manifest) as f:
            manifest = json.load(f)
        for s in manifest.get("samples", []):
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
                x1, y1 = max(0, x - 2), max(0, y - 2)
                x2, y2 = min(iw, x + w + 2), min(ih, y + h + 2)
                if x2 > x1 + 4 and y2 > y1 + 4:
                    word_crops.append((img[y1:y2, x1:x2], wd.get("text", "")))
    else:
        logger.warning("No manifest; generating synthetic crops...")
        gen = SyntheticPageGenerator(seed=cfg.seed, img_size=256)
        for _ in range(30):
            page = gen.generate_page()
            word_crops.extend(generate_word_crops(page["words"], page["image"]))

    if not word_crops:
        logger.error("No word crops. Run generate_synthetic_data.py first.")
        sys.exit(1)
    logger.info(f"SR pairs: {len(word_crops)}")

    sr_ds = SRPairDataset(
        word_crops,
        hr_size=(cfg.diffusion_sr.hr_height, cfg.diffusion_sr.hr_width),
        lr_size=(cfg.diffusion_sr.lr_height, cfg.diffusion_sr.lr_width),
    )
    loader = DataLoader(sr_ds, batch_size=cfg.diffusion_sr.batch_size,
                        shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    # ── Model + Schedule ───────────────────────────────────────────────────
    model = OCRDiffSR(
        base_ch=cfg.diffusion_sr.base_channels,
        time_dim=cfg.diffusion_sr.time_emb_dim,
        T=cfg.diffusion_sr.T,
    ).to(device)
    schedule = DiffusionSchedule(T=cfg.diffusion_sr.T, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.diffusion_sr.lr_pretrain)

    run_dir = os.path.join(cfg.output_dir, "diffusion_sr")
    ckptr = Checkpointer(run_dir)
    metrics_path = os.path.join(run_dir, "pretrain_metrics.jsonl")

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        state = Checkpointer.load(args.resume, model, optimizer, device)
        start_step = state.get("step", 0)

    # ── Training Loop ──────────────────────────────────────────────────────
    model.train()
    loss_meter = AverageMeter("mse_loss")
    max_iter = cfg.diffusion_sr.max_iterations
    logger.info(f"Pretraining diffusion SR: {len(sr_ds)} pairs, batch={cfg.diffusion_sr.batch_size}, max_iter={max_iter}")

    data_iter = iter(loader)
    step = start_step
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

        # Sample random timesteps
        t = torch.randint(0, cfg.diffusion_sr.T, (B,), device=device, dtype=torch.long)
        # Forward diffusion
        hr_scaled = hr  # already in [-1, 1]
        x_t, noise = schedule.q_sample(hr_scaled, t)

        optimizer.zero_grad()
        pred_noise = model(x_t, lr_up, t)
        loss = torch.nn.functional.mse_loss(pred_noise, noise)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        loss_meter.update(loss.item(), B)
        step += 1

        if step % 50 == 0:
            elapsed = time.time() - t0
            logger.info(f"[{step:>6}/{max_iter}] mse={loss_meter.avg:.6f} elapsed={elapsed:.0f}s")
            save_metrics({"step": step, "mse_loss": loss_meter.avg}, metrics_path)
            loss_meter.reset()

        if step % 500 == 0 or step == max_iter:
            ckptr.save(f"step{step}", model, optimizer, step)

    logger.info("Diffusion SR pretraining complete.")
    ckptr.save("pretrain_final", model, optimizer, step)


if __name__ == "__main__":
    main()
