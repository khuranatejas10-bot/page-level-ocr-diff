"""
End-to-end page OCR inference pipeline.
Orchestrates: preprocessing → detection → CC extraction → layout → recognise → postprocess.
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

from .preprocessing import preprocess_page, normalize_for_detector, normalize_word_crop, extract_word_crop
from .components import extract_components, TextComponent
from .grouping import build_layout, TextBlock, TextLine
from .reading_order import determine_reading_order, extract_ordered_text
from .postprocess import postprocess_page_text


class PageOCRResult:
    """Holds full page OCR output."""
    def __init__(self):
        self.text: str = ""
        self.lines: List[str] = []
        self.blocks: List[TextBlock] = []
        self.word_boxes: List[Dict] = []
        self.confidence: float = 0.0
        self.timing: Dict[str, float] = {}

    def __repr__(self):
        return f"PageOCRResult(lines={len(self.lines)}, confidence={self.confidence:.3f})"


class PageOCRPipeline:
    """
    Integrated page OCR pipeline (Paper 1 full system).

    Stages:
      1. Preprocess image
      2. Run FourHeadDetector
      3. Extract CCs with orientation/scale
      4. Layout: block grouping (Delaunay, epsilon=2) + line grouping (polynomial)
      5. Reading order (XY-cut + poly baseline sort)
      6. Optional OCR-Diff enhancement on low-confidence crops
      7. Run CTC recognizer on word crops
      8. Postprocess (WordNinja + confidence filter)
    """

    def __init__(
        self,
        detector,
        recognizer,
        vocab,
        diffusion_sr=None,
        config=None,
        device: str = "cpu",
    ):
        self.detector = detector.to(device).eval()
        self.recognizer = recognizer.to(device).eval()
        self.vocab = vocab
        self.diffusion_sr = diffusion_sr.to(device).eval() if diffusion_sr else None
        self.config = config
        self.device = device

        # Config shortcuts
        cfg_d = config.detector if config else None
        self.region_thresh = cfg_d.region_threshold if cfg_d else 0.4
        self.affinity_thresh = cfg_d.affinity_threshold if cfg_d else 0.4
        self.min_comp_area = cfg_d.min_component_area if cfg_d else 10
        self.min_comp_score = cfg_d.min_component_score if cfg_d else 0.3
        self.epsilon = cfg_d.delaunay_epsilon if cfg_d else 2.0
        self.max_poly_degree = cfg_d.max_poly_degree if cfg_d else 4
        self.dev_factor = cfg_d.line_deviation_scale_factor if cfg_d else 0.25
        self.scale_bin_edges = cfg_d.scale_bin_edges if cfg_d else None

        cfg_r = config.recognizer if config else None
        self.img_height = cfg_r.img_height if cfg_r else 32
        self.img_width = cfg_r.img_width if cfg_r else 128

        cfg_sr = config.diffusion_sr if config else None
        self.sr_conf_thresh = cfg_sr.enhancement_confidence_threshold if cfg_sr else 0.5
        self.sr_ddim_steps = cfg_sr.ddim_steps if cfg_sr else 20
        self.hr_size = (cfg_sr.hr_height, cfg_sr.hr_width) if cfg_sr else (32, 128)

    @torch.no_grad()
    def run(self, image: np.ndarray) -> PageOCRResult:
        """
        Run end-to-end OCR on a BGR or RGB image.
        Returns PageOCRResult with full text, structured blocks/lines, and timing.
        """
        result = PageOCRResult()
        t0 = time.time()

        # ── Stage 1: Preprocess ──────────────────────────────────────────────
        img_proc, scale = preprocess_page(image)
        img_norm = normalize_for_detector(img_proc)  # (H, W, 3) float32
        H, W = img_proc.shape[:2]
        result.timing["preprocess"] = time.time() - t0

        # ── Stage 2: Detect ──────────────────────────────────────────────────
        t1 = time.time()
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        det_out = self.detector(img_tensor)
        region_map = det_out["region"][0, 0].cpu().numpy()
        affinity_map = det_out["affinity"][0, 0].cpu().numpy()
        orient_logits = det_out["orientation"][0].cpu().numpy()  # (32, H/2, W/2)
        scale_logits = det_out["scale"][0].cpu().numpy()         # (10, H/2, W/2)
        result.timing["detection"] = time.time() - t1

        # ── Stage 3: Extract CCs ─────────────────────────────────────────────
        t2 = time.time()
        components = extract_components(
            region_map, affinity_map, orient_logits, scale_logits,
            region_thresh=self.region_thresh, affinity_thresh=self.affinity_thresh,
            min_area=self.min_comp_area, min_score=self.min_comp_score,
            scale_bin_edges=self.scale_bin_edges,
        )
        result.timing["cc_extraction"] = time.time() - t2

        if not components:
            result.text = ""
            return result

        # ── Stage 4 & 5: Layout + Reading Order ──────────────────────────────
        t3 = time.time()
        blocks = build_layout(components, epsilon=self.epsilon,
                              max_poly_degree=self.max_poly_degree,
                              deviation_scale_factor=self.dev_factor)
        blocks = determine_reading_order(blocks, page_width=W)
        result.blocks = blocks
        result.timing["layout"] = time.time() - t3

        # ── Stage 6 & 7: Recognize words ─────────────────────────────────────
        t4 = time.time()
        all_line_texts = []
        all_line_confs = []
        word_boxes_out = []

        for block in blocks:
            for line in block.lines:
                line_words = []
                line_conf_vals = []
                for comp in line.components:
                    bx, by, bw, bh = comp.bbox
                    # Map back to original image coords
                    crop = extract_word_crop(img_proc, (bx, by, bw, bh))
                    if crop.size == 0:
                        continue

                    # Optional: OCR-Diff SR enhancement
                    crop_rgb = crop[:, :, ::-1] if crop.ndim == 3 else crop
                    if self.diffusion_sr and comp.score < self.sr_conf_thresh:
                        crop_rgb = self._enhance_crop(crop_rgb)

                    crop_norm = normalize_word_crop(
                        crop[:, :, ::-1] if crop.ndim == 3 and crop.shape[2] == 3 else crop,
                        target_h=self.img_height
                    )
                    # Pad to max width
                    cw = crop_norm.shape[1]
                    if cw > self.img_width:
                        crop_norm = crop_norm[:, :self.img_width]
                        cw = self.img_width
                    pad_w = self.img_width - cw
                    if pad_w > 0:
                        crop_norm = np.pad(crop_norm, ((0, 0), (0, pad_w)),
                                          mode="constant", constant_values=-1.0)

                    crop_tensor = torch.from_numpy(crop_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
                    logits = self.recognizer(crop_tensor)  # (T, 1, C)
                    decoded = self.recognizer.decode_greedy(logits)
                    word_text = self.vocab.decode(decoded[0]) if decoded else ""

                    # Confidence: mean max-softmax over T
                    probs = torch.softmax(logits[:, 0, :], dim=-1)
                    conf = float(probs.max(dim=-1).values.mean())

                    if word_text.strip():
                        line_words.append(word_text)
                        line_conf_vals.append(conf)
                        word_boxes_out.append({
                            "text": word_text, "confidence": conf,
                            "bbox": [bx, by, bw, bh],
                            "block_id": block.id,
                        })

                if line_words:
                    all_line_texts.append(" ".join(line_words))
                    all_line_confs.append(float(np.mean(line_conf_vals)) if line_conf_vals else 0.0)

        result.timing["recognition"] = time.time() - t4

        # ── Stage 8: Postprocess ──────────────────────────────────────────────
        full_text, filtered_lines = postprocess_page_text(
            all_line_texts, all_line_confs
        )
        result.text = full_text
        result.lines = filtered_lines
        result.word_boxes = word_boxes_out
        result.confidence = float(np.mean(all_line_confs)) if all_line_confs else 0.0
        result.timing["total"] = time.time() - t0

        return result

    @torch.no_grad()
    def _enhance_crop(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Run OCR-Diff SR on a single low-confidence crop."""
        if self.diffusion_sr is None:
            return crop_rgb
        h, w = crop_rgb.shape[:2]
        crop_tensor = torch.from_numpy(
            crop_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        ).unsqueeze(0).to(self.device)
        enhanced = self.diffusion_sr.enhance(crop_tensor, self.hr_size, self.sr_ddim_steps)
        enhanced_np = (enhanced[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        return enhanced_np
