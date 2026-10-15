#!/usr/bin/env python
"""
scripts/infer.py
Run end-to-end inference on a single image or a directory of images.
Usage:
    python scripts/infer.py --image path/to/image.jpg \
        --det_ckpt runs/detector/ckpt_final.pt \
        --rec_ckpt runs/recognizer/ckpt_final.pt \
        --output_dir reports/infer
    python scripts/infer.py --image_dir path/to/images/ --output_dir reports/infer
"""
import argparse
import os
import sys
import json
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import torch

from src.pageocrdiff.config import Config
from src.pageocrdiff.models import FourHeadDetector, CTCRecognizer, Vocabulary
from src.pageocrdiff.pipeline import PageOCRPipeline
from src.pageocrdiff.utils import get_logger, visualise_detections

logger = get_logger("infer")


def parse_args():
    p = argparse.ArgumentParser(description="Run OCR inference on image(s)")
    p.add_argument("--config", default="configs/smoke_test.yaml")
    p.add_argument("--image", default=None, help="Single image path")
    p.add_argument("--image_dir", default=None, help="Directory of images")
    p.add_argument("--det_ckpt", default=None)
    p.add_argument("--rec_ckpt", default=None)
    p.add_argument("--sr_ckpt", default=None)
    p.add_argument("--output_dir", default="reports/infer")
    p.add_argument("--save_vis", action="store_true", help="Save visualisations")
    return p.parse_args()


def build_pipeline(args, cfg, device):
    detector = FourHeadDetector(
        backbone=cfg.detector.backbone, pretrained=False,
        num_orient=cfg.detector.num_orientation_bins,
        num_scale=cfg.detector.num_scale_bins,
    )
    if args.det_ckpt and os.path.exists(args.det_ckpt):
        state = torch.load(args.det_ckpt, map_location="cpu", weights_only=False)
        detector.load_state_dict(state["model"])
        logger.info(f"Loaded detector: {args.det_ckpt}")

    vocab = Vocabulary(cfg.recognizer.charset)
    recognizer = CTCRecognizer(num_classes=vocab.num_classes)
    if args.rec_ckpt and os.path.exists(args.rec_ckpt):
        state = torch.load(args.rec_ckpt, map_location="cpu", weights_only=False)
        recognizer.load_state_dict(state["model"])
        logger.info(f"Loaded recognizer: {args.rec_ckpt}")

    sr_model = None
    if args.sr_ckpt and os.path.exists(args.sr_ckpt):
        from src.pageocrdiff.models import OCRDiffSR
        sr_model = OCRDiffSR(base_ch=cfg.diffusion_sr.base_channels,
                              time_dim=cfg.diffusion_sr.time_emb_dim,
                              T=cfg.diffusion_sr.T)
        state = torch.load(args.sr_ckpt, map_location="cpu", weights_only=False)
        sr_model.load_state_dict(state["model"])

    return PageOCRPipeline(detector, recognizer, vocab,
                           diffusion_sr=sr_model, config=cfg, device=device)


def run_on_image(pipeline, img_path: str, output_dir: str, save_vis: bool):
    img = cv2.imread(img_path)
    if img is None:
        logger.warning(f"Cannot read: {img_path}")
        return

    result = pipeline.run(img)

    name = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Save text
    txt_path = os.path.join(output_dir, f"{name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result.text)

    # Save JSON
    json_path = os.path.join(output_dir, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump({
            "text": result.text,
            "lines": result.lines,
            "confidence": result.confidence,
            "num_words": len(result.word_boxes),
            "timing": result.timing,
        }, f, indent=2)

    # Save visualisation
    if save_vis:
        vis = visualise_detections(img, result.word_boxes)
        vis_path = os.path.join(output_dir, f"{name}_vis.png")
        cv2.imwrite(vis_path, vis)

    logger.info(f"  {name}: {len(result.lines)} lines, conf={result.confidence:.3f} → {txt_path}")


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config) if os.path.exists(args.config) else Config()
    device = cfg.resolve_device()
    pipeline = build_pipeline(args, cfg, device)

    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.image_dir:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_paths = sorted(image_paths)
    else:
        logger.error("Provide --image or --image_dir")
        sys.exit(1)

    logger.info(f"Running inference on {len(image_paths)} image(s) → {args.output_dir}")
    for img_path in image_paths:
        run_on_image(pipeline, img_path, args.output_dir, args.save_vis)

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
