#!/usr/bin/env python
"""
scripts/run_smoke_test.py
End-to-end smoke test: generates tiny synthetic data, runs 5 training iterations
of each stage, runs inference on 2 pages, and evaluates all metrics.
Completes in ~60-120s on CPU. Used to verify the full system is runnable.

Usage:
    python scripts/run_smoke_test.py
    python scripts/run_smoke_test.py --quick   # Even smaller test
"""
import argparse
import os
import sys
import time
import json
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.pageocrdiff.config import Config
from src.pageocrdiff.utils import set_seed, get_logger

logger = get_logger("smoke_test")

SMOKE_DATA_DIR = "data/smoke"
SMOKE_RUN_DIR = "runs/smoke"
SMOKE_REPORT = "reports/smoke_test_report.json"


def parse_args():
    p = argparse.ArgumentParser(description="Run end-to-end smoke test")
    p.add_argument("--quick", action="store_true", help="Minimal iteration counts")
    return p.parse_args()


def make_smoke_config(quick: bool) -> Config:
    cfg = Config()
    cfg.seed = 42
    cfg.data_root = SMOKE_DATA_DIR
    cfg.output_dir = SMOKE_RUN_DIR
    cfg.num_workers = 0
    cfg.detector.image_size = 256        # Smaller for speed
    cfg.detector.batch_size = 2
    cfg.detector.max_iterations = 5 if quick else 10
    cfg.detector.pretrained_backbone = False  # Avoid download in CI
    cfg.detector.lr_decay_every = 1000
    cfg.recognizer.batch_size = 4
    cfg.recognizer.max_iterations = 5 if quick else 10
    cfg.diffusion_sr.batch_size = 2
    cfg.diffusion_sr.max_iterations = 5 if quick else 10
    cfg.diffusion_sr.T = 100             # Short schedule for smoke test
    cfg.diffusion_sr.base_channels = 16  # Tiny U-Net
    cfg.diffusion_sr.time_emb_dim = 32
    cfg.diffusion_sr.fe_channels = 16
    cfg.diffusion_sr.ddim_steps = 5
    return cfg


def step_generate_data(cfg: Config, n: int = 8) -> bool:
    logger.info("\n── STEP 1: Generate Synthetic Data ────────────────────────")
    try:
        from src.pageocrdiff.data.synthetic_page_generator import SyntheticPageGenerator
        gen = SyntheticPageGenerator(seed=cfg.seed, img_size=cfg.detector.image_size)
        for split, count in [("train", n), ("val", max(2, n//4)), ("test", max(2, n//4))]:
            manifest = gen.generate_dataset(cfg.data_root, count, split=split)
            logger.info(f"  Generated {count} {split} samples → {manifest}")
        return True
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        traceback.print_exc()
        return False


def step_inspect(cfg: Config) -> bool:
    logger.info("\n── STEP 2: Inspect Dataset ─────────────────────────────────")
    try:
        manifest_path = os.path.join(cfg.data_root, "train_manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        samples = manifest.get("samples", [])
        missing = sum(1 for s in samples
                      if not os.path.exists(os.path.join(cfg.data_root, s["image"])))
        logger.info(f"  Train samples: {len(samples)}, missing images: {missing}")
        assert missing == 0, f"{missing} images missing"
        return True
    except Exception as e:
        logger.error(f"Inspection failed: {e}")
        traceback.print_exc()
        return False


def step_train_detector(cfg: Config) -> bool:
    logger.info("\n── STEP 3: Train Detector ──────────────────────────────────")
    try:
        from src.pageocrdiff.models import FourHeadDetector, DetectorLoss
        from src.pageocrdiff.data import DetectorDataset
        from torch.utils.data import DataLoader

        train_manifest = os.path.join(cfg.data_root, "train_manifest.json")
        ds = DetectorDataset(train_manifest, cfg.data_root,
                             img_size=cfg.detector.image_size, augment=False)
        loader = DataLoader(ds, batch_size=cfg.detector.batch_size,
                            shuffle=True, num_workers=0)

        detector = FourHeadDetector(
            backbone=cfg.detector.backbone,
            pretrained=cfg.detector.pretrained_backbone,
            num_orient=cfg.detector.num_orientation_bins,
            num_scale=cfg.detector.num_scale_bins,
        )
        criterion = DetectorLoss()
        optimizer = torch.optim.Adam(detector.parameters(), lr=1e-4)

        detector.train()
        data_iter = iter(loader)
        for step in range(cfg.detector.max_iterations):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            images = batch["image"]
            targets = {k: batch[k] for k in
                       ["region_gt", "affinity_gt", "orientation_gt", "scale_gt", "text_mask"]}
            optimizer.zero_grad()
            preds = detector(images)
            losses = criterion(preds, targets)
            losses["total"].backward()
            optimizer.step()

        # Save checkpoint
        os.makedirs(SMOKE_RUN_DIR + "/detector", exist_ok=True)
        torch.save({"step": cfg.detector.max_iterations, "model": detector.state_dict()},
                   SMOKE_RUN_DIR + "/detector/ckpt_final.pt")
        logger.info(f"  Detector trained {cfg.detector.max_iterations} steps. Loss: {losses['total'].item():.4f}")
        return True
    except Exception as e:
        logger.error(f"Detector training failed: {e}")
        traceback.print_exc()
        return False


def step_train_recognizer(cfg: Config) -> bool:
    logger.info("\n── STEP 4: Train Recognizer ────────────────────────────────")
    try:
        from src.pageocrdiff.models import CTCRecognizer, Vocabulary
        from src.pageocrdiff.data import RecognizerDataset, recognizer_collate
        from src.pageocrdiff.data.synthetic_page_generator import SyntheticPageGenerator, generate_word_crops
        from torch.utils.data import DataLoader
        import json, cv2

        vocab = Vocabulary(cfg.recognizer.charset)

        # Collect word crops from generated data
        manifest_path = os.path.join(cfg.data_root, "train_manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        word_crops = []
        for s in manifest.get("samples", []):
            img = cv2.imread(os.path.join(cfg.data_root, s["image"]))
            if img is None:
                continue
            anno_path = os.path.join(cfg.data_root, s["annotation"])
            with open(anno_path) as f:
                anno = json.load(f)
            ih, iw = img.shape[:2]
            for wd in anno.get("words", []):
                if not wd.get("text"):
                    continue
                x, y, w, h = int(wd["x"]), int(wd["y"]), int(wd["w"]), int(wd["h"])
                crop = img[max(0, y-2):min(ih, y+h+2), max(0, x-2):min(iw, x+w+2)]
                if crop.shape[0] > 4 and crop.shape[1] > 4:
                    word_crops.append((crop, wd["text"]))

        if not word_crops:
            # Fallback: generate a few directly
            gen = SyntheticPageGenerator(seed=cfg.seed, img_size=cfg.detector.image_size)
            for _ in range(5):
                page = gen.generate_page()
                word_crops.extend(generate_word_crops(page["words"], page["image"]))

        ds = RecognizerDataset(word_crops[:max(8, len(word_crops))], vocab,
                                img_height=cfg.recognizer.img_height,
                                img_width=cfg.recognizer.img_width)
        loader = DataLoader(ds, batch_size=cfg.recognizer.batch_size, shuffle=True,
                            num_workers=0, collate_fn=recognizer_collate, drop_last=False)

        recognizer = CTCRecognizer(num_classes=vocab.num_classes)
        ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        optimizer = torch.optim.Adam(recognizer.parameters(), lr=1e-4)

        recognizer.train()
        data_iter = iter(loader)
        final_loss = float("nan")
        for step in range(cfg.recognizer.max_iterations):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            images = batch["image"]
            labels = batch["label"]
            label_lens = batch["label_len"]
            optimizer.zero_grad()
            logits = recognizer(images)
            T, B, C = logits.shape
            input_lens = torch.full((B,), T, dtype=torch.long)
            loss = ctc_loss(logits.log_softmax(2), labels, input_lens, label_lens)
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                final_loss = loss.item()

        os.makedirs(SMOKE_RUN_DIR + "/recognizer", exist_ok=True)
        torch.save({"step": cfg.recognizer.max_iterations, "model": recognizer.state_dict()},
                   SMOKE_RUN_DIR + "/recognizer/ckpt_final.pt")
        # Save vocab
        with open(SMOKE_RUN_DIR + "/recognizer/vocab.json", "w") as f:
            json.dump({"charset": cfg.recognizer.charset}, f)
        logger.info(f"  Recognizer trained {cfg.recognizer.max_iterations} steps. "
                    f"Last CTC loss: {final_loss:.4f}")
        return True
    except Exception as e:
        logger.error(f"Recognizer training failed: {e}")
        traceback.print_exc()
        return False


def step_train_diffusion(cfg: Config) -> bool:
    logger.info("\n── STEP 5: Train Diffusion SR ──────────────────────────────")
    try:
        from src.pageocrdiff.models import OCRDiffSR
        from src.pageocrdiff.models.diffusion_sr import DiffusionSchedule
        from src.pageocrdiff.data import SRPairDataset
        from src.pageocrdiff.data.synthetic_page_generator import SyntheticPageGenerator, generate_word_crops
        from torch.utils.data import DataLoader
        import json, cv2

        manifest_path = os.path.join(cfg.data_root, "train_manifest.json")
        word_crops = []
        with open(manifest_path) as f:
            manifest = json.load(f)
        for s in manifest.get("samples", []):
            img = cv2.imread(os.path.join(cfg.data_root, s["image"]))
            if img is None:
                continue
            anno_path = os.path.join(cfg.data_root, s["annotation"])
            with open(anno_path) as f:
                anno = json.load(f)
            ih, iw = img.shape[:2]
            for wd in anno.get("words", []):
                x, y, w, h = int(wd["x"]), int(wd["y"]), int(wd["w"]), int(wd["h"])
                crop = img[max(0, y-2):min(ih, y+h+2), max(0, x-2):min(iw, x+w+2)]
                if crop.shape[0] > 4 and crop.shape[1] > 4:
                    word_crops.append((crop, wd.get("text", "")))
        if not word_crops:
            gen = SyntheticPageGenerator(seed=cfg.seed, img_size=cfg.detector.image_size)
            for _ in range(5):
                page = gen.generate_page()
                word_crops.extend(generate_word_crops(page["words"], page["image"]))

        ds = SRPairDataset(word_crops,
                           hr_size=(cfg.diffusion_sr.hr_height, cfg.diffusion_sr.hr_width),
                           lr_size=(cfg.diffusion_sr.lr_height, cfg.diffusion_sr.lr_width))
        loader = DataLoader(ds, batch_size=cfg.diffusion_sr.batch_size,
                            shuffle=True, num_workers=0, drop_last=False)

        model = OCRDiffSR(base_ch=cfg.diffusion_sr.base_channels,
                           time_dim=cfg.diffusion_sr.time_emb_dim, T=cfg.diffusion_sr.T)
        schedule = DiffusionSchedule(T=cfg.diffusion_sr.T)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        data_iter = iter(loader)
        final_loss = float("nan")
        for step in range(cfg.diffusion_sr.max_iterations):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            hr = batch["hr"]
            lr_up = batch["lr_up"]
            B = hr.shape[0]
            t = torch.randint(0, cfg.diffusion_sr.T, (B,), dtype=torch.long)
            x_t, noise = schedule.q_sample(hr, t)
            optimizer.zero_grad()
            pred = model(x_t, lr_up, t)
            loss = torch.nn.functional.mse_loss(pred, noise)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        os.makedirs(SMOKE_RUN_DIR + "/diffusion_sr", exist_ok=True)
        torch.save({"step": cfg.diffusion_sr.max_iterations, "model": model.state_dict()},
                   SMOKE_RUN_DIR + "/diffusion_sr/ckpt_final.pt")
        logger.info(f"  Diffusion SR trained {cfg.diffusion_sr.max_iterations} steps. MSE: {final_loss:.6f}")
        return True
    except Exception as e:
        logger.error(f"Diffusion SR training failed: {e}")
        traceback.print_exc()
        return False


def step_inference_and_evaluate(cfg: Config) -> dict:
    logger.info("\n── STEP 6: Inference + Evaluate ────────────────────────────")
    results = {}
    try:
        from src.pageocrdiff.models import FourHeadDetector, CTCRecognizer, Vocabulary
        from src.pageocrdiff.pipeline import PageOCRPipeline
        from src.pageocrdiff.metrics import evaluate_page
        import json, cv2

        vocab = Vocabulary(cfg.recognizer.charset)
        detector = FourHeadDetector(
            backbone=cfg.detector.backbone, pretrained=False,
            num_orient=cfg.detector.num_orientation_bins,
            num_scale=cfg.detector.num_scale_bins,
        )
        det_ckpt = SMOKE_RUN_DIR + "/detector/ckpt_final.pt"
        if os.path.exists(det_ckpt):
            state = torch.load(det_ckpt, map_location="cpu", weights_only=False)
            detector.load_state_dict(state["model"])

        recognizer = CTCRecognizer(num_classes=vocab.num_classes)
        rec_ckpt = SMOKE_RUN_DIR + "/recognizer/ckpt_final.pt"
        if os.path.exists(rec_ckpt):
            state = torch.load(rec_ckpt, map_location="cpu", weights_only=False)
            recognizer.load_state_dict(state["model"])

        pipeline = PageOCRPipeline(detector, recognizer, vocab, config=cfg, device="cpu")

        manifest_path = os.path.join(cfg.data_root, "test_manifest.json")
        if not os.path.exists(manifest_path):
            manifest_path = os.path.join(cfg.data_root, "val_manifest.json")

        with open(manifest_path) as f:
            manifest = json.load(f)
        samples = manifest.get("samples", [])[:2]

        page_metrics = []
        for s in samples:
            img = cv2.imread(os.path.join(cfg.data_root, s["image"]))
            if img is None:
                continue
            with open(os.path.join(cfg.data_root, s["annotation"])) as f:
                anno = json.load(f)
            result = pipeline.run(img)
            m = evaluate_page(result.text, anno.get("transcription", ""))
            m["id"] = s["id"]
            page_metrics.append(m)
            logger.info(f"  {s['id']}: CER={m['cer']:.4f}  WER={m['wer']:.4f}  timing={result.timing}")

        if page_metrics:
            results["mean_cer"] = float(np.mean([m["cer"] for m in page_metrics]))
            results["mean_wer"] = float(np.mean([m["wer"] for m in page_metrics]))
            results["pages_evaluated"] = len(page_metrics)

        logger.info(f"  Mean CER={results.get('mean_cer', 0):.4f}  "
                    f"WER={results.get('mean_wer', 0):.4f}")
        return results
    except Exception as e:
        logger.error(f"Inference/evaluation failed: {e}")
        traceback.print_exc()
        results["error"] = str(e)
        return results


def step_run_unit_tests() -> bool:
    logger.info("\n── STEP 7: Unit Tests ──────────────────────────────────────")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short", "--no-header"],
            capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr
        logger.info(output[-3000:] if len(output) > 3000 else output)
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"pytest not available or tests failed: {e}")
        return False


def main():
    args = parse_args()
    set_seed(42)
    cfg = make_smoke_config(args.quick)

    t_total = time.time()
    report = {
        "quick": args.quick,
        "steps": {},
        "eval_metrics": {},
    }

    logger.info("=" * 60)
    logger.info("PageOCRDiff Smoke Test")
    logger.info("=" * 60)

    report["steps"]["generate_data"] = step_generate_data(cfg)
    report["steps"]["inspect"]       = step_inspect(cfg)
    report["steps"]["train_detector"]    = step_train_detector(cfg)
    report["steps"]["train_recognizer"]  = step_train_recognizer(cfg)
    report["steps"]["train_diffusion"]   = step_train_diffusion(cfg)
    report["eval_metrics"]               = step_inference_and_evaluate(cfg)
    report["steps"]["unit_tests"]        = step_run_unit_tests()

    elapsed = time.time() - t_total
    report["elapsed_seconds"] = round(elapsed, 1)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 60)
    all_pass = all(v for v in report["steps"].values() if isinstance(v, bool))
    for name, passed in report["steps"].items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}  {name}")
    logger.info(f"\nTotal time: {elapsed:.1f}s")
    logger.info(f"Result: {'ALL PASS ✓' if all_pass else 'SOME STEPS FAILED ✗'}")

    os.makedirs(os.path.dirname(SMOKE_REPORT) or ".", exist_ok=True)
    with open(SMOKE_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report: {SMOKE_REPORT}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
