#!/usr/bin/env python
"""
scripts/inspect_dataset.py
Stage 0: Inspect a dataset manifest — validate images/targets, print statistics.
Usage:
    python scripts/inspect_dataset.py --manifest data/synthetic/train_manifest.json \
        --data_root data/synthetic --visualise --n_samples 5
"""
import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pageocrdiff.utils import get_logger

logger = get_logger("inspect_dataset")


def parse_args():
    p = argparse.ArgumentParser(description="Inspect OCR dataset manifest")
    p.add_argument("--manifest", required=True, help="Path to *_manifest.json")
    p.add_argument("--data_root", default="data/synthetic", help="Dataset root dir")
    p.add_argument("--visualise", action="store_true", help="Save sample visualisations")
    p.add_argument("--n_samples", type=int, default=5, help="Number of samples to inspect")
    p.add_argument("--out_dir", default="reports/inspect", help="Visualisation output dir")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    samples = manifest.get("samples", [])
    logger.info(f"Manifest: {args.manifest}")
    logger.info(f"  Split:   {manifest.get('split', '?')}")
    logger.info(f"  Samples: {len(samples)}")

    missing_images = 0
    missing_targets = 0
    region_max_vals = []
    num_words_list = []

    for s in samples:
        img_p = os.path.join(args.data_root, s["image"])
        tgt_p = os.path.join(args.data_root, s["targets"])
        anno_p = os.path.join(args.data_root, s["annotation"])

        if not os.path.exists(img_p):
            missing_images += 1
        if not os.path.exists(tgt_p):
            missing_targets += 1
            continue

        try:
            tgt = np.load(tgt_p)
            region_max_vals.append(float(tgt["region"].max()))
        except Exception as e:
            logger.warning(f"  Cannot load targets for {s['id']}: {e}")

        if os.path.exists(anno_p):
            with open(anno_p) as f:
                anno = json.load(f)
            num_words_list.append(len(anno.get("words", [])))

    logger.info(f"\n── Validation ──────────────────────────")
    logger.info(f"  Missing images:  {missing_images}/{len(samples)}")
    logger.info(f"  Missing targets: {missing_targets}/{len(samples)}")
    if region_max_vals:
        logger.info(f"  Region max (mean): {np.mean(region_max_vals):.4f}")
        logger.info(f"  Region max (min/max): {min(region_max_vals):.4f} / {max(region_max_vals):.4f}")
    if num_words_list:
        logger.info(f"  Words/page (mean): {np.mean(num_words_list):.1f}")
        logger.info(f"  Words/page (min/max): {min(num_words_list)} / {max(num_words_list)}")

    if args.visualise:
        import cv2
        os.makedirs(args.out_dir, exist_ok=True)
        for s in samples[:args.n_samples]:
            img_p = os.path.join(args.data_root, s["image"])
            tgt_p = os.path.join(args.data_root, s["targets"])
            img = cv2.imread(img_p)
            if img is None:
                continue
            tgt = np.load(tgt_p)
            region = (tgt["region"] * 255).astype(np.uint8)
            region_vis = cv2.applyColorMap(region, cv2.COLORMAP_HOT)
            h_out = min(img.shape[0], 384)
            scale = h_out / img.shape[0]
            img_sm = cv2.resize(img, (int(img.shape[1] * scale), h_out))
            reg_sm = cv2.resize(region_vis, (region_vis.shape[1] * 2, region_vis.shape[0]))
            reg_sm = cv2.resize(reg_sm, (img_sm.shape[1], h_out))
            vis = np.hstack([img_sm, reg_sm])
            out_path = os.path.join(args.out_dir, f"{s['id']}_inspect.png")
            cv2.imwrite(out_path, vis)
            logger.info(f"  Saved: {out_path}")

    logger.info("Inspection complete.")


if __name__ == "__main__":
    main()
