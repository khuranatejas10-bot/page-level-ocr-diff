#!/usr/bin/env python
"""
scripts/train_pipeline.py
End-to-end training orchestrator for PageOCRDiff on Kaggle / real OCR datasets.

Executes:
  1. Data Preparation (Kaggle download or high-fidelity document generation)
  2. Four-Head Detector Training (Region, Affinity, Orientation, Scale)
  3. CTC Word Recognizer Training
  4. OCR-Diff Diffusion Super-Resolution Training
  5. Full Pipeline Evaluation (CER, WER, Detection P/R/F1)

Usage:
    python scripts/train_pipeline.py --config configs/kaggle_train.yaml
    python scripts/train_pipeline.py --source kaggle --dataset abhishek0189/standard-ocr-dataset
"""
import argparse
import os
import sys
import time
import json
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_command_stream(cmd, desc):
    print("\n" + "=" * 65)
    print(f"STEP: {desc}")
    print(f"COMMAND: {' '.join(cmd)}")
    print("=" * 65)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"\n[Error] {desc} failed with exit code {proc.returncode} ({elapsed:.1f}s)")
        sys.exit(proc.returncode)
    print(f"\n[Completed] {desc} in {elapsed:.1f}s")
    return elapsed


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end OCR model training pipeline")
    p.add_argument("--config", default="configs/kaggle_train.yaml", help="Path to config YAML")
    p.add_argument("--source", default="auto", choices=["auto", "kaggle", "augmented"],
                   help="Dataset source: auto, kaggle, or augmented")
    p.add_argument("--dataset", default="abhishek0189/standard-ocr-dataset",
                   help="Kaggle dataset slug")
    p.add_argument("--kaggle-username", default=None, help="Kaggle username")
    p.add_argument("--kaggle-key", default=None, help="Kaggle API key")
    p.add_argument("--skip-data", action="store_true", help="Skip dataset preparation if already prepared")
    p.add_argument("--skip-detector", action="store_true", help="Skip detector training")
    p.add_argument("--skip-recognizer", action="store_true", help="Skip recognizer training")
    p.add_argument("--skip-diffusion", action="store_true", help="Skip diffusion SR training")
    p.add_argument("--skip-eval", action="store_true", help="Skip pipeline evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    total_start = time.time()

    print("*" * 65)
    print("PageOCRDiff: End-to-End OCR Training Pipeline")
    print(f"Configuration: {args.config}")
    print(f"Dataset Source: {args.source}")
    print("*" * 65)

    py = sys.executable

    # 1. Prepare Dataset
    if not args.skip_data:
        data_cmd = [
            py, "scripts/download_kaggle_dataset.py",
            "--source", args.source,
            "--dataset", args.dataset,
            "--output-dir", "data/kaggle",
            "--img-size", "256",
            "--train-pages", "40",
            "--val-pages", "10",
            "--test-pages", "10"
        ]
        if args.kaggle_username and args.kaggle_key:
            data_cmd.extend(["--kaggle-username", args.kaggle_username, "--kaggle-key", args.kaggle_key])
        run_command_stream(data_cmd, "1/5: Dataset Preparation")
    else:
        print("\n[Skip] Step 1/5: Dataset Preparation skipped by user flag")

    # 2. Train Detector
    if not args.skip_detector:
        run_command_stream(
            [py, "scripts/train_detector.py", "--config", args.config],
            "2/5: Train Four-Head Detector"
        )
    else:
        print("\n[Skip] Step 2/5: Detector Training skipped")

    # 3. Train Recognizer
    if not args.skip_recognizer:
        run_command_stream(
            [py, "scripts/train_recognizer.py", "--config", args.config],
            "3/5: Train CTC Word Recognizer"
        )
    else:
        print("\n[Skip] Step 3/5: Recognizer Training skipped")

    # 4. Train Diffusion SR
    if not args.skip_diffusion:
        run_command_stream(
            [py, "scripts/train_diffusion_sr.py", "--config", args.config],
            "4/5: Train Diffusion Super-Resolution"
        )
    else:
        print("\n[Skip] Step 4/5: Diffusion SR Training skipped")

    # 5. Evaluate Pipeline
    if not args.skip_eval:
        eval_cmd = [
            py, "scripts/evaluate.py",
            "--config", args.config,
            "--det_ckpt", "runs/kaggle_trained/detector/ckpt_final.pt",
            "--rec_ckpt", "runs/kaggle_trained/recognizer/ckpt_final.pt",
            "--sr_ckpt", "runs/kaggle_trained/diffusion_sr/ckpt_final.pt",
            "--manifest", "data/kaggle/test_manifest.json",
            "--output", "runs/kaggle_trained/evaluation_results.json"
        ]
        run_command_stream(eval_cmd, "5/5: Pipeline Evaluation")
    else:
        print("\n[Skip] Step 5/5: Evaluation skipped")

    total_time = time.time() - total_start
    print("\n" + "*" * 65)
    print(f"TRAINING PIPELINE FINISHED IN {total_time/60:.2f} MINUTES")
    print("Trained checkpoints saved at:")
    print("  * Detector:     runs/kaggle_trained/detector/ckpt_final.pt")
    print("  * Recognizer:   runs/kaggle_trained/recognizer/ckpt_final.pt")
    print("  * Diffusion SR: runs/kaggle_trained/diffusion_sr/ckpt_final.pt")
    if not args.skip_eval and os.path.exists("runs/kaggle_trained/evaluation_results.json"):
        print("  * Metrics:      runs/kaggle_trained/evaluation_results.json")
        try:
            with open("runs/kaggle_trained/evaluation_results.json") as f:
                res = json.load(f)
            print("\nEvaluation Summary:")
            for k, v in res.get("metrics", {}).items():
                print(f"    - {k}: {v}")
        except Exception:
            pass
    print("*" * 65)


if __name__ == "__main__":
    main()
