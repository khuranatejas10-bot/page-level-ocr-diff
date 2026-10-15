#!/usr/bin/env python
"""
cli.py - Unified Command-Line Interface for PageOCRDiff.
Synthesizes IEEE Access 2025 and IEEE IoTJ 2024.

Usage:
    # Run end-to-end OCR on an image
    python cli.py --image path/to/image.jpg --output results.txt --save-vis vis.png

    # Run smoke test
    python cli.py smoke-test

    # Generate synthetic dataset
    python cli.py generate-data --train 200 --val 40 --test 40

    # Train stages
    python cli.py train-detector --config configs/smoke_test.yaml
    python cli.py train-recognizer --config configs/smoke_test.yaml
    python cli.py train-diffusion --config configs/smoke_test.yaml

    # Evaluate
    python cli.py evaluate --config configs/smoke_test.yaml
"""
import argparse
import os
import sys
import json
import cv2

# Ensure project root is on path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def cmd_infer(args):
    """Run inference on an image or directory of images."""
    from src.pageocrdiff.config import Config
    from src.pageocrdiff.models import FourHeadDetector, CTCRecognizer, Vocabulary
    from src.pageocrdiff.pipeline import PageOCRPipeline
    from src.pageocrdiff.metrics import cer, wer, tbrl
    from src.pageocrdiff.utils import visualise_detections, get_logger
    import torch

    logger = get_logger("cli_infer")

    cfg_path = args.config if hasattr(args, "config") and args.config else "configs/smoke_test.yaml"
    cfg = Config.from_yaml(cfg_path) if os.path.exists(cfg_path) else Config()
    device = cfg.resolve_device()

    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        return 1

    img = cv2.imread(args.image)
    if img is None:
        logger.error(f"Cannot load image: {args.image}")
        return 1

    # Initialize models
    vocab = Vocabulary(cfg.recognizer.charset)
    detector = FourHeadDetector(
        backbone=cfg.detector.backbone, pretrained=False,
        num_orient=cfg.detector.num_orientation_bins,
        num_scale=cfg.detector.num_scale_bins,
    )
    det_ckpt = getattr(args, "det_ckpt", None) or "runs/smoke/detector/ckpt_final.pt"
    if os.path.exists(det_ckpt):
        st = torch.load(det_ckpt, map_location="cpu", weights_only=False)
        detector.load_state_dict(st["model"])
        logger.info(f"Loaded detector weights: {det_ckpt}")

    recognizer = CTCRecognizer(num_classes=vocab.num_classes)
    rec_ckpt = getattr(args, "rec_ckpt", None) or "runs/smoke/recognizer/ckpt_final.pt"
    if os.path.exists(rec_ckpt):
        st = torch.load(rec_ckpt, map_location="cpu", weights_only=False)
        recognizer.load_state_dict(st["model"])
        logger.info(f"Loaded recognizer weights: {rec_ckpt}")

    sr_model = None
    use_diffusion = not getattr(args, "no_diffusion", False)
    if use_diffusion:
        sr_ckpt = getattr(args, "sr_ckpt", None) or "runs/smoke/diffusion_sr/ckpt_final.pt"
        if os.path.exists(sr_ckpt):
            from src.pageocrdiff.models import OCRDiffSR
            sr_model = OCRDiffSR(base_ch=cfg.diffusion_sr.base_channels,
                                  time_dim=cfg.diffusion_sr.time_emb_dim,
                                  T=cfg.diffusion_sr.T)
            st = torch.load(sr_ckpt, map_location="cpu", weights_only=False)
            sr_model.load_state_dict(st["model"])
            logger.info(f"Loaded diffusion SR weights: {sr_ckpt}")

    pipeline = PageOCRPipeline(detector, recognizer, vocab,
                               diffusion_sr=sr_model, config=cfg, device=device)

    print(f"\n[PageOCRDiff] Processing document: {args.image}")
    result = pipeline.run(img)

    print("\n" + "=" * 60)
    print("EXTRACTED DOCUMENT TEXT (IN TOPOLOGICAL READING ORDER):")
    print("=" * 60)
    print(result.text if result.text.strip() else "[No high-confidence text detected]")
    print("=" * 60)
    print(f"Confidence: {result.confidence:.3f} | Timing: {result.timing}")

    # Ground truth evaluation if provided
    gt_path = getattr(args, "gt", None)
    if gt_path and os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_text = f.read()
        c = cer(result.text, gt_text)
        w = wer(result.text, gt_text)
        pred_b = [b for b in result.text.split("\n\n") if b.strip()]
        gt_b = [b for b in gt_text.split("\n\n") if b.strip()]
        t = tbrl(pred_b, gt_b)
        print(f"\nEvaluation Metrics vs GT:")
        print(f"  CER:   {c:.4f}")
        print(f"  WER:   {w:.4f}")
        print(f"  TBR-L: {t:.4f}")

    if getattr(args, "save_vis", None):
        vis = visualise_detections(img, result.word_boxes)
        cv2.imwrite(args.save_vis, vis)
        print(f"Visualization saved to: {args.save_vis}")

    out_path = getattr(args, "output", None)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if out_path.endswith(".json"):
            data = {
                "text": result.text,
                "lines": result.lines,
                "confidence": result.confidence,
                "timing": result.timing,
                "word_boxes": result.word_boxes,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(result.text)
        print(f"Output saved to: {out_path}")

    return 0


def cmd_smoke_test(args):
    import subprocess
    cmd = [sys.executable, "scripts/run_smoke_test.py"]
    if getattr(args, "quick", False):
        cmd.append("--quick")
    return subprocess.run(cmd).returncode


def cmd_generate_data(args):
    import subprocess
    cmd = [
        sys.executable, "scripts/generate_synthetic_data.py",
        "--output", getattr(args, "output", "data/synthetic"),
        "--train", str(getattr(args, "train", 200)),
        "--val", str(getattr(args, "val", 40)),
        "--test", str(getattr(args, "test", 40)),
        "--img_size", str(getattr(args, "img_size", 768)),
        "--seed", str(getattr(args, "seed", 42)),
    ]
    return subprocess.run(cmd).returncode


def cmd_train_detector(args):
    import subprocess
    cmd = [sys.executable, "scripts/train_detector.py", "--config", getattr(args, "config", "configs/default.yaml")]
    if getattr(args, "max_iter", None):
        cmd.extend(["--max_iter", str(args.max_iter)])
    return subprocess.run(cmd).returncode


def cmd_train_recognizer(args):
    import subprocess
    cmd = [sys.executable, "scripts/train_recognizer.py", "--config", getattr(args, "config", "configs/default.yaml")]
    if getattr(args, "max_iter", None):
        cmd.extend(["--max_iter", str(args.max_iter)])
    return subprocess.run(cmd).returncode


def cmd_train_diffusion(args):
    import subprocess
    cmd = [sys.executable, "scripts/train_diffusion_sr.py", "--config", getattr(args, "config", "configs/default.yaml")]
    if getattr(args, "max_iter", None):
        cmd.extend(["--max_iter", str(args.max_iter)])
    return subprocess.run(cmd).returncode


def cmd_evaluate(args):
    import subprocess
    cmd = [sys.executable, "scripts/evaluate.py", "--config", getattr(args, "config", "configs/smoke_test.yaml")]
    if getattr(args, "manifest", None):
        cmd.extend(["--manifest", args.manifest])
    if getattr(args, "output", None):
        cmd.extend(["--output", args.output])
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(
        description="PageOCRDiff: Page-Level OCR with Diffusion Super-Resolution (IEEE Access 2025 + IEEE IoTJ 2024)"
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Subcommand: smoke-test
    p_smoke = subparsers.add_parser("smoke-test", help="Run end-to-end smoke test")
    p_smoke.add_argument("--quick", action="store_true", default=True, help="Fast iteration count")

    # Subcommand: generate-data
    p_data = subparsers.add_parser("generate-data", help="Generate synthetic OCR dataset")
    p_data.add_argument("--output", default="data/synthetic")
    p_data.add_argument("--train", type=int, default=200)
    p_data.add_argument("--val", type=int, default=40)
    p_data.add_argument("--test", type=int, default=40)
    p_data.add_argument("--img_size", type=int, default=768)
    p_data.add_argument("--seed", type=int, default=42)

    # Subcommand: train-detector
    p_tdet = subparsers.add_parser("train-detector", help="Train four-head detector")
    p_tdet.add_argument("--config", default="configs/default.yaml")
    p_tdet.add_argument("--max_iter", type=int, default=None)

    # Subcommand: train-recognizer
    p_trec = subparsers.add_parser("train-recognizer", help="Train CTC recognizer")
    p_trec.add_argument("--config", default="configs/default.yaml")
    p_trec.add_argument("--max_iter", type=int, default=None)

    # Subcommand: train-diffusion
    p_tdif = subparsers.add_parser("train-diffusion", help="Pretrain diffusion SR")
    p_tdif.add_argument("--config", default="configs/default.yaml")
    p_tdif.add_argument("--max_iter", type=int, default=None)

    # Subcommand: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate full pipeline")
    p_eval.add_argument("--config", default="configs/smoke_test.yaml")
    p_eval.add_argument("--manifest", default=None)
    p_eval.add_argument("--output", default="reports/evaluation_results.json")

    # Subcommand: infer (also handles top-level --image)
    p_infer = subparsers.add_parser("infer", help="Run document OCR inference")
    for p in (parser, p_infer):
        p.add_argument("--image", default=None, help="Path to input document image")
        p.add_argument("--config", default="configs/smoke_test.yaml")
        p.add_argument("--output", default=None, help="Path to output text or JSON file")
        p.add_argument("--save-vis", dest="save_vis", default=None, help="Path to save visualization")
        p.add_argument("--gt", default=None, help="Path to ground truth text file for evaluation")
        p.add_argument("--no-diffusion", action="store_true", help="Disable diffusion SR")
        p.add_argument("--det-ckpt", dest="det_ckpt", default=None)
        p.add_argument("--rec-ckpt", dest="rec_ckpt", default=None)
        p.add_argument("--sr-ckpt", dest="sr_ckpt", default=None)

    args = parser.parse_args()

    if args.subcommand == "smoke-test":
        return cmd_smoke_test(args)
    elif args.subcommand == "generate-data":
        return cmd_generate_data(args)
    elif args.subcommand == "train-detector":
        return cmd_train_detector(args)
    elif args.subcommand == "train-recognizer":
        return cmd_train_recognizer(args)
    elif args.subcommand == "train-diffusion":
        return cmd_train_diffusion(args)
    elif args.subcommand == "evaluate":
        return cmd_evaluate(args)
    elif args.subcommand == "infer" or args.image:
        if not args.image:
            parser.error("Please specify --image for inference.")
        return cmd_infer(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
