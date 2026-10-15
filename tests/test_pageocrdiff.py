"""
Unit tests for PageOCRDiff — all stages.
Run with: python -m pytest tests/ -q
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch


# ─────────────────────────── Metrics tests ────────────────────────────────

class TestEditDistance:
    def test_identical(self):
        from src.pageocrdiff.metrics import edit_distance
        assert edit_distance("hello", "hello") == 0

    def test_empty(self):
        from src.pageocrdiff.metrics import edit_distance
        assert edit_distance("", "abc") == 3
        assert edit_distance("abc", "") == 3

    def test_substitution(self):
        from src.pageocrdiff.metrics import edit_distance
        assert edit_distance("kitten", "sitting") == 3

    def test_cer_zero(self):
        from src.pageocrdiff.metrics import cer
        assert cer("hello world", "hello world") == 0.0

    def test_cer_normalized(self):
        from src.pageocrdiff.metrics import cer
        v = cer("abc", "xyz")
        assert 0.0 <= v <= 1.5  # Can exceed 1 with insertions

    def test_wer(self):
        from src.pageocrdiff.metrics import wer
        assert wer("the cat sat", "the cat sat") == 0.0
        assert wer("", "hello") > 0.0


class TestIoU:
    def test_perfect(self):
        from src.pageocrdiff.metrics import iou_box
        assert iou_box((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_no_overlap(self):
        from src.pageocrdiff.metrics import iou_box
        assert iou_box((0, 0, 5, 5), (10, 10, 5, 5)) == pytest.approx(0.0)

    def test_partial(self):
        from src.pageocrdiff.metrics import iou_box
        v = iou_box((0, 0, 10, 10), (5, 5, 10, 10))
        assert 0.0 < v < 1.0


class TestDetectionPRF:
    def test_perfect(self):
        from src.pageocrdiff.metrics import detection_prf
        boxes = [(0, 0, 10, 10), (20, 20, 10, 10)]
        m = detection_prf(boxes, boxes, iou_threshold=0.5)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)

    def test_empty_preds(self):
        from src.pageocrdiff.metrics import detection_prf
        m = detection_prf([], [(0, 0, 10, 10)], 0.5)
        assert m["precision"] == 0.0 and m["recall"] == 0.0


class TestTBRL:
    def test_perfect(self):
        from src.pageocrdiff.metrics import tbrl
        blocks = ["block A", "block B", "block C"]
        assert tbrl(blocks, blocks) == pytest.approx(1.0)

    def test_empty(self):
        from src.pageocrdiff.metrics import tbrl
        assert tbrl([], ["a"]) == 0.0

    def test_permuted(self):
        from src.pageocrdiff.metrics import tbrl
        ref = ["A", "B", "C"]
        pred = ["C", "A", "B"]
        v = tbrl(pred, ref)
        assert 0.0 < v < 1.0


class TestPSNR:
    def test_identical(self):
        from src.pageocrdiff.metrics import psnr
        img = np.random.rand(32, 32, 3).astype(np.float32)
        assert psnr(img, img) == float("inf")

    def test_noisy(self):
        from src.pageocrdiff.metrics import psnr
        img = np.ones((32, 32), dtype=np.float32) * 0.5
        noisy = img + np.random.normal(0, 0.05, img.shape).astype(np.float32)
        v = psnr(img, noisy)
        assert 10 < v < 50


# ─────────────────────────── Model tests ──────────────────────────────────

class TestFourHeadDetector:
    def test_output_shapes(self):
        from src.pageocrdiff.models import FourHeadDetector
        model = FourHeadDetector(backbone="resnet18", pretrained=False,
                                 num_orient=32, num_scale=10)
        model.eval()
        x = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out["region"].shape == (1, 1, 128, 128)
        assert out["affinity"].shape == (1, 1, 128, 128)
        assert out["orientation"].shape == (1, 32, 128, 128)
        assert out["scale"].shape == (1, 10, 128, 128)

    def test_region_in_0_1(self):
        from src.pageocrdiff.models import FourHeadDetector
        model = FourHeadDetector(backbone="resnet18", pretrained=False)
        model.eval()
        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out["region"].min() >= -1e-4
        assert out["region"].max() <= 1.0 + 1e-4

    def test_detector_loss(self):
        from src.pageocrdiff.models import FourHeadDetector, DetectorLoss
        model = FourHeadDetector(pretrained=False)
        criterion = DetectorLoss()
        x = torch.rand(2, 3, 128, 128)
        preds = model(x)
        H, W = preds["region"].shape[2], preds["region"].shape[3]
        targets = {
            "region_gt": torch.rand(2, 1, H, W),
            "affinity_gt": torch.rand(2, 1, H, W),
            "orientation_gt": torch.randint(0, 32, (2, H, W)),
            "scale_gt": torch.randint(0, 10, (2, H, W)),
            "text_mask": torch.ones(2, H, W),
        }
        losses = criterion(preds, targets)
        assert torch.isfinite(losses["total"])
        assert losses["total"] > 0


class TestCTCRecognizer:
    def test_output_shape(self):
        from src.pageocrdiff.models import CTCRecognizer
        model = CTCRecognizer(num_classes=97)
        x = torch.zeros(2, 1, 32, 128)
        logits = model(x)
        T, B, C = logits.shape
        assert B == 2 and C == 97 and T > 0

    def test_greedy_decode(self):
        from src.pageocrdiff.models import CTCRecognizer
        model = CTCRecognizer(num_classes=97)
        model.eval()
        x = torch.zeros(1, 1, 32, 128)
        with torch.no_grad():
            logits = model(x)
        decoded = model.decode_greedy(logits)
        assert isinstance(decoded, list)
        assert len(decoded) == 1

    def test_vocabulary(self):
        from src.pageocrdiff.models import Vocabulary
        vocab = Vocabulary("abcABC0123")
        enc = vocab.encode("abc")
        assert len(enc) == 3
        dec = vocab.decode(enc)
        assert dec == "abc"


class TestOCRDiffSR:
    def test_forward(self):
        from src.pageocrdiff.models import OCRDiffSR
        model = OCRDiffSR(base_ch=16, time_dim=32, T=50)
        x_t = torch.rand(1, 3, 32, 128)
        x_up = torch.rand(1, 3, 32, 128)
        t = torch.tensor([10])
        pred = model(x_t, x_up, t)
        assert pred.shape == (1, 3, 32, 128)

    def test_diffusion_schedule(self):
        from src.pageocrdiff.models.diffusion_sr import DiffusionSchedule
        schedule = DiffusionSchedule(T=50)
        x0 = torch.rand(2, 3, 16, 16)
        t = torch.tensor([10, 30])
        x_t, noise = schedule.q_sample(x0, t)
        assert x_t.shape == x0.shape
        assert noise.shape == x0.shape


# ─────────────────────────── Data tests ───────────────────────────────────

class TestSyntheticGenerator:
    def test_generate_page(self):
        from src.pageocrdiff.data import SyntheticPageGenerator
        gen = SyntheticPageGenerator(seed=42, img_size=256)
        page = gen.generate_page(apply_augmentation=False)
        assert page["image"].shape == (256, 256, 3)
        assert isinstance(page["words"], list)
        assert isinstance(page["transcription"], str)
        assert "region" in page["targets"]
        assert page["targets"]["region"].shape == (128, 128)

    def test_target_maps_range(self):
        from src.pageocrdiff.data import SyntheticPageGenerator
        gen = SyntheticPageGenerator(seed=0, img_size=128)
        page = gen.generate_page(apply_augmentation=False)
        t = page["targets"]
        assert t["region"].min() >= 0.0 and t["region"].max() <= 1.0
        assert t["orientation"].min() >= 0
        assert t["scale"].min() >= 0

    def test_word_crops(self):
        from src.pageocrdiff.data import SyntheticPageGenerator, generate_word_crops
        gen = SyntheticPageGenerator(seed=1, img_size=256)
        page = gen.generate_page(apply_augmentation=False)
        crops = generate_word_crops(page["words"], page["image"])
        assert len(crops) > 0
        crop, text = crops[0]
        assert crop.shape[0] == 32  # target height
        assert isinstance(text, str)


# ─────────────────────────── Pipeline tests ───────────────────────────────

class TestPreprocessing:
    def test_preprocess_page(self):
        from src.pageocrdiff.pipeline import preprocess_page
        img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        proc, scale = preprocess_page(img, target_size=256, do_deskew=False)
        assert proc.shape[2] == 3
        assert proc.shape[0] % 32 == 0
        assert proc.shape[1] % 32 == 0
        assert scale > 0

    def test_normalize_word_crop(self):
        from src.pageocrdiff.pipeline import normalize_word_crop
        crop = np.random.randint(100, 200, (24, 80, 3), dtype=np.uint8)
        out = normalize_word_crop(crop, target_h=32)
        assert out.shape[0] == 32
        assert out.dtype == np.float32


class TestGrouping:
    def test_build_layout_single(self):
        from src.pageocrdiff.pipeline.components import TextComponent
        from src.pageocrdiff.pipeline.grouping import build_layout
        comps = [
            TextComponent(id=0, center=(50.0, 50.0), bbox=(40, 40, 20, 15),
                          area=300, orientation_deg=0.0, orientation_bin=0,
                          scale=20.0, scale_bin=2, score=0.8)
        ]
        blocks = build_layout(comps)
        assert len(blocks) == 1

    def test_delaunay_grouping(self):
        from src.pageocrdiff.pipeline.components import TextComponent
        from src.pageocrdiff.pipeline.grouping import group_components_into_blocks
        comps = [
            TextComponent(id=i, center=(float(i * 30), 50.0),
                          bbox=(i * 30 - 15, 40, 28, 15),
                          area=300, orientation_deg=0.0, orientation_bin=0,
                          scale=20.0, scale_bin=2, score=0.8)
            for i in range(5)
        ]
        blocks = group_components_into_blocks(comps, epsilon=2.0)
        assert len(blocks) >= 1


class TestReadingOrder:
    def test_no_blocks(self):
        from src.pageocrdiff.pipeline.reading_order import determine_reading_order
        result = determine_reading_order([])
        assert result == []


class TestPostprocess:
    def test_clean_text(self):
        from src.pageocrdiff.pipeline.postprocess import clean_ocr_text
        text = "  hello   world  \n\n  foo  "
        clean = clean_ocr_text(text)
        assert "hello world" in clean

    def test_confidence_filter(self):
        from src.pageocrdiff.pipeline.postprocess import filter_low_confidence
        texts = ["hello", "world", "junk"]
        confs = [0.9, 0.8, 0.05]
        filtered = filter_low_confidence(texts, confs, threshold=0.3)
        # 'junk' should be filtered (confidence too low)
        assert len(filtered) < 3


# ─────────────────────────── Integration test ─────────────────────────────

class TestEndToEnd:
    """Minimal end-to-end test: generate one page and run pipeline."""
    def test_pipeline_runs(self):
        from src.pageocrdiff.data import SyntheticPageGenerator
        from src.pageocrdiff.models import FourHeadDetector, CTCRecognizer, Vocabulary
        from src.pageocrdiff.pipeline import PageOCRPipeline
        from src.pageocrdiff.config import Config

        cfg = Config()
        cfg.detector.pretrained_backbone = False
        cfg.detector.image_size = 128
        cfg.diffusion_sr.T = 10
        cfg.diffusion_sr.base_channels = 8
        cfg.diffusion_sr.time_emb_dim = 16

        vocab = Vocabulary(cfg.recognizer.charset)
        detector = FourHeadDetector(pretrained=False,
                                     num_orient=cfg.detector.num_orientation_bins,
                                     num_scale=cfg.detector.num_scale_bins)
        recognizer = CTCRecognizer(num_classes=vocab.num_classes)
        pipeline = PageOCRPipeline(detector, recognizer, vocab, config=cfg, device="cpu")

        gen = SyntheticPageGenerator(seed=99, img_size=128)
        page = gen.generate_page(apply_augmentation=False)
        import cv2
        img_bgr = cv2.cvtColor(page["image"], cv2.COLOR_RGB2BGR)
        result = pipeline.run(img_bgr)

        assert isinstance(result.text, str)
        assert isinstance(result.lines, list)
        assert isinstance(result.confidence, float)
        assert "total" in result.timing
