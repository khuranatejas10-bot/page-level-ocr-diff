#!/usr/bin/env python
"""
scripts/train_detector.py
Stage 1: Train the four-head text detector.
Usage:
    python scripts/train_detector.py --config configs/smoke_test.yaml
    python scripts/train_detector.py --config configs/default.yaml --max_iter 50000
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from src.pageocrdiff.config import Config
from src.pageocrdiff.models import FourHeadDetector, DetectorLoss
from src.pageocrdiff.data import DetectorDataset
from src.pageocrdiff.utils import set_seed, get_logger, AverageMeter, Checkpointer, save_metrics

logger = get_logger("train_detector")


def parse_args():
    p = argparse.ArgumentParser(description="Train four-head detector")
    p.add_argument("--config", default="configs/smoke_test.yaml")
    p.add_argument("--max_iter", type=int, default=None, help="Override max_iterations")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--output_dir", default=None, help="Override output directory")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config.from_yaml(args.config) if os.path.exists(args.config) else Config()
    if args.max_iter:
        cfg.detector.max_iterations = args.max_iter
    if args.output_dir:
        cfg.output_dir = args.output_dir

    device = cfg.resolve_device()
    set_seed(cfg.seed)
    logger.info(f"Device: {device} | Config: {args.config}")

    # ── Data ───────────────────────────────────────────────────────────────
    data_root = cfg.data_root
    train_manifest = os.path.join(data_root, "train_manifest.json")
    val_manifest = os.path.join(data_root, "val_manifest.json")

    if not os.path.exists(train_manifest):
        logger.error(f"Train manifest not found: {train_manifest}")
        logger.error("Run: python scripts/generate_synthetic_data.py first.")
        sys.exit(1)

    train_ds = DetectorDataset(train_manifest, data_root,
                               img_size=cfg.detector.image_size, augment=True)
    val_ds = DetectorDataset(val_manifest, data_root,
                             img_size=cfg.detector.image_size, augment=False) if os.path.exists(val_manifest) else None

    train_loader = DataLoader(train_ds, batch_size=cfg.detector.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              drop_last=True)

    # ── Model ──────────────────────────────────────────────────────────────
    model = FourHeadDetector(
        backbone=cfg.detector.backbone,
        pretrained=cfg.detector.pretrained_backbone,
        num_orient=cfg.detector.num_orientation_bins,
        num_scale=cfg.detector.num_scale_bins,
    ).to(device)

    criterion = DetectorLoss(
        lambda_scale=cfg.detector.lambda_scale,
        lambda_orient=cfg.detector.lambda_orientation,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.detector.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.detector.lr_decay_every,
        gamma=cfg.detector.lr_decay_factor,
    )

    run_dir = os.path.join(cfg.output_dir, "detector")
    ckptr = Checkpointer(run_dir)
    metrics_path = os.path.join(run_dir, "train_metrics.jsonl")

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        state = Checkpointer.load(args.resume, model, optimizer, device)
        start_step = state.get("step", 0)
        logger.info(f"Resumed from step {start_step}")

    # ── Training Loop ──────────────────────────────────────────────────────
    model.train()
    loss_meter = AverageMeter("loss")
    region_meter = AverageMeter("region")
    affinity_meter = AverageMeter("affinity")
    t0 = time.time()
    step = start_step
    max_iter = cfg.detector.max_iterations

    logger.info(f"Training detector: {len(train_ds)} samples, "
                f"batch={cfg.detector.batch_size}, max_iter={max_iter}")

    data_iter = iter(train_loader)
    while step < max_iter:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        images = batch["image"].to(device)
        targets = {
            "region_gt": batch["region_gt"].to(device),
            "affinity_gt": batch["affinity_gt"].to(device),
            "orientation_gt": batch["orientation_gt"].to(device),
            "scale_gt": batch["scale_gt"].to(device),
            "text_mask": batch["text_mask"].to(device),
        }

        optimizer.zero_grad()
        preds = model(images)
        losses = criterion(preds, targets)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        B = images.shape[0]
        loss_meter.update(losses["total"].item(), B)
        region_meter.update(losses["region"].item(), B)
        affinity_meter.update(losses["affinity"].item(), B)
        step += 1

        if step % 50 == 0:
            elapsed = time.time() - t0
            logger.info(f"[{step:>6}/{max_iter}] loss={loss_meter.avg:.4f} "
                        f"region={region_meter.avg:.4f} affinity={affinity_meter.avg:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"elapsed={elapsed:.0f}s")
            save_metrics({"step": step, "loss": loss_meter.avg,
                          "region": region_meter.avg, "affinity": affinity_meter.avg},
                         metrics_path)
            loss_meter.reset(); region_meter.reset(); affinity_meter.reset()

        if step % 500 == 0 or step == max_iter:
            ckptr.save(f"step{step}", model, optimizer, step,
                       {"loss": loss_meter.avg if loss_meter.count > 0 else 0.0})

    logger.info("Detector training complete.")
    ckptr.save("final", model, optimizer, step)


if __name__ == "__main__":
    main()
