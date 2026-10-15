#!/usr/bin/env python
"""
scripts/generate_synthetic_data.py
Generate a local synthetic dataset for smoke-test / offline training.
Usage:
    python scripts/generate_synthetic_data.py --output data/synthetic \
        --train 200 --val 40 --test 40 --img_size 768 --seed 42
"""
import argparse
import os
import sys
import time

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pageocrdiff.data.synthetic_page_generator import SyntheticPageGenerator
from src.pageocrdiff.utils import get_logger

logger = get_logger("generate_data")


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic OCR dataset")
    p.add_argument("--output", default="data/synthetic", help="Output root directory")
    p.add_argument("--train", type=int, default=200, help="Number of training samples")
    p.add_argument("--val", type=int, default=40, help="Number of validation samples")
    p.add_argument("--test", type=int, default=40, help="Number of test samples")
    p.add_argument("--img_size", type=int, default=768, help="Image size (longest side)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    gen = SyntheticPageGenerator(seed=args.seed, img_size=args.img_size)

    splits = [("train", args.train), ("val", args.val), ("test", args.test)]
    for split, n in splits:
        logger.info(f"Generating {n} {split} samples → {args.output}/{split}/")
        t0 = time.time()
        gen.rng.seed(args.seed + {"train": 0, "val": 1000, "test": 2000}[split])
        manifest = gen.generate_dataset(args.output, n, split=split)
        elapsed = time.time() - t0
        logger.info(f"  Done in {elapsed:.1f}s. Manifest: {manifest}")

    logger.info(f"Dataset generation complete. Root: {args.output}")


if __name__ == "__main__":
    main()
