#!/usr/bin/env python
"""
scripts/train_recognizer.py
Stage 2: Train the CTC word recognizer.
Usage:
    python scripts/train_recognizer.py --config configs/smoke_test.yaml
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader
import json

from src.pageocrdiff.config import Config
from src.pageocrdiff.models import CTCRecognizer, Vocabulary
from src.pageocrdiff.data.synthetic_page_generator import SyntheticPageGenerator, generate_word_crops
from src.pageocrdiff.data import RecognizerDataset, recognizer_collate
from src.pageocrdiff.utils import set_seed, get_logger, AverageMeter, Checkpointer, save_metrics

logger = get_logger("train_recognizer")


def parse_args():
    p = argparse.ArgumentParser(description="Train CTC recognizer")
    p.add_argument("--config", default="configs/smoke_test.yaml")
    p.add_argument("--max_iter", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config) if os.path.exists(args.config) else Config()
    if args.max_iter:
        cfg.recognizer.max_iterations = args.max_iter
    if args.output_dir:
        cfg.output_dir = args.output_dir

    device = cfg.resolve_device()
    set_seed(cfg.seed)
    logger.info(f"Device: {device}")

    # ── Build vocabulary ───────────────────────────────────────────────────
    vocab = Vocabulary(cfg.recognizer.charset)
    logger.info(f"Vocabulary size: {vocab.num_classes} classes (incl. blank)")

    # ── Gather word crops from synthetic data ──────────────────────────────
    data_root = cfg.data_root
    train_manifest = os.path.join(data_root, "train_manifest.json")

    word_crops = []
    if os.path.exists(train_manifest):
        with open(train_manifest) as f:
            manifest = json.load(f)
        samples = manifest.get("samples", [])
        import cv2
        for s in samples:
            img_path = os.path.join(data_root, s["image"])
            anno_path = os.path.join(data_root, s["annotation"])
            img = cv2.imread(img_path)
            if img is None:
                continue
            if not os.path.exists(anno_path):
                continue
            with open(anno_path) as f:
                anno = json.load(f)
            for wd in anno.get("words", []):
                if not wd.get("text"):
                    continue
                x, y, w, h = int(wd["x"]), int(wd["y"]), int(wd["w"]), int(wd["h"])
                ih, iw = img.shape[:2]
                x1, y1 = max(0, x - 2), max(0, y - 2)
                x2, y2 = min(iw, x + w + 2), min(ih, y + h + 2)
                if x2 > x1 and y2 > y1:
                    crop = img[y1:y2, x1:x2]
                    word_crops.append((crop, wd["text"]))
    else:
        logger.warning("No train manifest found; generating synthetic word crops...")
        gen = SyntheticPageGenerator(seed=cfg.seed, img_size=256)
        for _ in range(50):
            page = gen.generate_page()
            wcs = generate_word_crops(page["words"], page["image"])
            word_crops.extend(wcs)

    logger.info(f"Word crops: {len(word_crops)}")
    if not word_crops:
        logger.error("No word crops found. Run generate_synthetic_data.py first.")
        sys.exit(1)

    train_ds = RecognizerDataset(
        word_crops, vocab,
        img_height=cfg.recognizer.img_height,
        img_width=cfg.recognizer.img_width,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.recognizer.batch_size,
        shuffle=True, num_workers=cfg.num_workers,
        collate_fn=recognizer_collate, drop_last=True,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = CTCRecognizer(
        num_classes=vocab.num_classes,
        use_bilstm=cfg.recognizer.use_bilstm,
        bilstm_hidden=cfg.recognizer.bilstm_hidden,
    ).to(device)

    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.recognizer.lr)

    run_dir = os.path.join(cfg.output_dir, "recognizer")
    ckptr = Checkpointer(run_dir)
    metrics_path = os.path.join(run_dir, "train_metrics.jsonl")

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        state = Checkpointer.load(args.resume, model, optimizer, device)
        start_step = state.get("step", 0)

    # ── Training Loop ──────────────────────────────────────────────────────
    model.train()
    loss_meter = AverageMeter("ctc_loss")
    max_iter = cfg.recognizer.max_iterations
    logger.info(f"Training recognizer: {len(train_ds)} crops, batch={cfg.recognizer.batch_size}, max_iter={max_iter}")

    data_iter = iter(train_loader)
    step = start_step
    t0 = time.time()

    while step < max_iter:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        label_lens = batch["label_len"]

        optimizer.zero_grad()
        logits = model(images)  # (T, B, C)
        T, B, C = logits.shape
        input_lens = torch.full((B,), T, dtype=torch.long)
        loss = ctc_loss(
            logits.log_softmax(2).cpu(),
            labels.cpu(),
            input_lens,
            label_lens,
        )
        if not torch.isfinite(loss):
            step += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.recognizer.grad_clip)
        optimizer.step()
        loss_meter.update(loss.item(), B)
        step += 1

        if step % 50 == 0:
            elapsed = time.time() - t0
            logger.info(f"[{step:>6}/{max_iter}] ctc_loss={loss_meter.avg:.4f} elapsed={elapsed:.0f}s")
            save_metrics({"step": step, "ctc_loss": loss_meter.avg}, metrics_path)
            loss_meter.reset()

        if step % 500 == 0 or step == max_iter:
            ckptr.save(f"step{step}", model, optimizer, step)

    logger.info("Recognizer training complete.")
    ckptr.save("final", model, optimizer, step)

    # Save vocabulary
    vocab_path = os.path.join(run_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"charset": cfg.recognizer.charset}, f)
    logger.info(f"Vocabulary saved: {vocab_path}")


if __name__ == "__main__":
    main()
