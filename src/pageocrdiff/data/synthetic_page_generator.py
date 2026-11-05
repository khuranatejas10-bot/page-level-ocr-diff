"""
Synthetic document page generator for training and smoke tests.
Produces images with exact word/char boxes, transcriptions, and derived target maps
(region, affinity, orientation, scale) for the 4-head detector.
"""
import os
import json
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
import cv2


# ---- Default font handling ----
_SYSTEM_FONTS = [
    "arial.ttf", "times.ttf", "cour.ttf", "verdana.ttf", "calibri.ttf",
    "consola.ttf", "georgia.ttf", "tahoma.ttf",
]

def _find_fonts() -> List[str]:
    """Find available TrueType fonts on the system."""
    found = []
    search_dirs = []
    if os.name == "nt":
        search_dirs.append(os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
    search_dirs.extend(["/usr/share/fonts", "/usr/local/share/fonts"])
    for d in search_dirs:
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".ttf", ".otf")):
                        found.append(os.path.join(root, f))
    return found if found else []


# ---- Text content generation ----
_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "Document", "Image", "Recognition", "Camera", "Captured", "Page", "Level",
    "Optical", "Character", "Detection", "Network", "Deep", "Learning", "Model",
    "text", "line", "word", "block", "paragraph", "section", "chapter", "table",
    "figure", "header", "footer", "title", "subtitle", "caption", "reference",
    "The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "123", "456",
    "OCR", "system", "processing", "algorithm", "neural", "diffusion", "framework",
]


def _random_sentence(rng: random.Random, min_words: int = 3, max_words: int = 12) -> str:
    n = rng.randint(min_words, max_words)
    words = [rng.choice(_WORDS) for _ in range(n)]
    words[0] = words[0].capitalize()
    if rng.random() < 0.3:
        words.append(rng.choice([".", ",", "!", "?"]))
    return " ".join(words)


def _random_paragraph(rng: random.Random, min_lines: int = 2, max_lines: int = 6) -> List[str]:
    n = rng.randint(min_lines, max_lines)
    return [_random_sentence(rng) for _ in range(n)]


# ---- Target map generation ----

def _make_gaussian(size: int, sigma: float) -> np.ndarray:
    """Create a 2D Gaussian kernel."""
    x = np.arange(0, size, dtype=np.float32) - size // 2
    g = np.exp(-x**2 / (2 * sigma**2))
    return np.outer(g, g)


def render_region_map(char_boxes: List[dict], map_h: int, map_w: int,
                      stride: int = 2) -> np.ndarray:
    """
    Render CRAFT-style region score map from character bounding boxes.
    Each character center gets a Gaussian blob.
    """
    region = np.zeros((map_h, map_w), dtype=np.float32)
    for cb in char_boxes:
        cx = int(cb["cx"] / stride)
        cy = int(cb["cy"] / stride)
        r = max(3, int(min(cb["w"], cb["h"]) / stride / 2))
        g = _make_gaussian(2 * r + 1, r / 2.0)
        g = g / (g.max() + 1e-8)
        y1, y2 = max(0, cy - r), min(map_h, cy + r + 1)
        x1, x2 = max(0, cx - r), min(map_w, cx + r + 1)
        gy1 = r - (cy - y1)
        gy2 = gy1 + (y2 - y1)
        gx1 = r - (cx - x1)
        gx2 = gx1 + (x2 - x1)
        if y2 > y1 and x2 > x1 and gy2 > gy1 and gx2 > gx1:
            region[y1:y2, x1:x2] = np.maximum(region[y1:y2, x1:x2], g[gy1:gy2, gx1:gx2])
    return region


def render_affinity_map(char_boxes: List[dict], word_indices: List[Tuple[int, int]],
                        map_h: int, map_w: int, stride: int = 2) -> np.ndarray:
    """
    Render affinity score map: Gaussian at midpoints between consecutive characters
    within the same word.
    """
    affinity = np.zeros((map_h, map_w), dtype=np.float32)
    for start_idx, end_idx in word_indices:
        for i in range(start_idx, end_idx - 1):
            c1, c2 = char_boxes[i], char_boxes[i + 1]
            mx = int(((c1["cx"] + c2["cx"]) / 2) / stride)
            my = int(((c1["cy"] + c2["cy"]) / 2) / stride)
            r = max(2, int(min(c1["w"], c1["h"]) / stride / 3))
            g = _make_gaussian(2 * r + 1, r / 2.0)
            g = g / (g.max() + 1e-8)
            y1, y2 = max(0, my - r), min(map_h, my + r + 1)
            x1, x2 = max(0, mx - r), min(map_w, mx + r + 1)
            gy1, gy2 = r - (my - y1), r - (my - y1) + (y2 - y1)
            gx1, gx2 = r - (mx - x1), r - (mx - x1) + (x2 - x1)
            if y2 > y1 and x2 > x1 and gy2 > gy1 and gx2 > gx1:
                affinity[y1:y2, x1:x2] = np.maximum(affinity[y1:y2, x1:x2], g[gy1:gy2, gx1:gx2])
    return affinity


def render_orientation_map(char_boxes: List[dict], map_h: int, map_w: int,
                           num_bins: int = 32, stride: int = 2) -> np.ndarray:
    """Render per-pixel orientation class map (background = 0)."""
    orient_map = np.zeros((map_h, map_w), dtype=np.int64)
    for cb in char_boxes:
        angle_deg = cb.get("angle", 0.0) % 180.0
        bin_idx = int(angle_deg / (180.0 / num_bins))
        bin_idx = min(bin_idx, num_bins - 1)
        x1 = int(cb["x"] / stride)
        y1 = int(cb["y"] / stride)
        x2 = int((cb["x"] + cb["w"]) / stride)
        y2 = int((cb["y"] + cb["h"]) / stride)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(map_w, x2), min(map_h, y2)
        if y2 > y1 and x2 > x1:
            orient_map[y1:y2, x1:x2] = bin_idx
    return orient_map


def render_scale_map(char_boxes: List[dict], line_heights: Dict[int, float],
                     scale_bin_edges: List[float], map_h: int, map_w: int,
                     stride: int = 2) -> np.ndarray:
    """Render per-pixel scale class map based on interline distance."""
    scale_map = np.zeros((map_h, map_w), dtype=np.int64)
    for cb in char_boxes:
        line_id = cb.get("line_id", 0)
        scale_val = line_heights.get(line_id, cb["h"] * 1.5)
        bin_idx = 0
        for i in range(len(scale_bin_edges) - 1):
            if scale_val >= scale_bin_edges[i] and scale_val < scale_bin_edges[i + 1]:
                bin_idx = i
                break
        bin_idx = min(bin_idx, len(scale_bin_edges) - 2)
        x1, y1 = max(0, int(cb["x"] / stride)), max(0, int(cb["y"] / stride))
        x2, y2 = min(map_w, int((cb["x"] + cb["w"]) / stride)), min(map_h, int((cb["y"] + cb["h"]) / stride))
        if y2 > y1 and x2 > x1:
            scale_map[y1:y2, x1:x2] = bin_idx
    return scale_map


def render_text_mask(char_boxes: List[dict], map_h: int, map_w: int,
                     stride: int = 2) -> np.ndarray:
    """Binary mask of text-positive pixels at map resolution."""
    mask = np.zeros((map_h, map_w), dtype=np.float32)
    for cb in char_boxes:
        x1, y1 = max(0, int(cb["x"] / stride)), max(0, int(cb["y"] / stride))
        x2, y2 = min(map_w, int((cb["x"] + cb["w"]) / stride)), min(map_h, int((cb["y"] + cb["h"]) / stride))
        if y2 > y1 and x2 > x1:
            mask[y1:y2, x1:x2] = 1.0
    return mask


# ---- Page rendering ----

class SyntheticPageGenerator:
    """
    Generates synthetic document page images with exact annotations.
    Produces: image, word boxes, char boxes, transcriptions, and derived 4-tensor targets.
    """
    def __init__(self, seed: int = 42, img_size: int = 768,
                 scale_bin_edges: Optional[List[float]] = None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.img_size = img_size
        self.scale_bin_edges = scale_bin_edges or [0, 8, 12, 18, 26, 36, 48, 62, 78, 96, 1e6]
        self.fonts = _find_fonts()
        self._fallback_font = None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        if self.fonts:
            path = self.rng.choice(self.fonts[:min(8, len(self.fonts))])
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
        # Fallback to default
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def generate_page(self, apply_augmentation: bool = True) -> dict:
        """
        Generate one synthetic page with annotations.
        Returns dict with keys: 'image', 'words', 'char_boxes', 'word_indices',
        'lines', 'blocks', 'transcription', 'targets'.
        """
        W, H = self.img_size, self.img_size
        # Random background
        bg_val = self.rng.randint(230, 255)
        img = Image.new("RGB", (W, H), (bg_val, bg_val, bg_val))
        draw = ImageDraw.Draw(img)

        char_boxes = []
        word_list = []
        word_indices = []  # (start_char_idx, end_char_idx) for each word
        line_data = []
        block_data = []
        line_heights = {}

        # Layout: 1-2 columns, multiple blocks
        num_columns = self.rng.choice([1, 1, 1, 2])
        col_width = (W - 80) // num_columns
        block_id = 0
        line_id = 0

        for col in range(num_columns):
            col_x = 40 + col * (col_width + 20)
            y_cursor = 40 + self.rng.randint(0, 30)

            num_blocks = self.rng.randint(1, 3)
            for _ in range(num_blocks):
                if y_cursor > H - 60:
                    break
                font_size = self.rng.choice([14, 16, 18, 20, 22, 24])
                font = self._get_font(font_size)
                paragraph = _random_paragraph(self.rng, 2, 5)
                angle = 0.0  # Horizontal text for baseline
                if self.rng.random() < 0.15:
                    angle = self.rng.uniform(-5, 5)

                block_lines = []
                block_start_line = line_id

                for text_line in paragraph:
                    if y_cursor > H - 40:
                        break

                    line_x = col_x + self.rng.randint(0, 10)
                    line_chars_start = len(char_boxes)
                    line_words_start = len(word_list)

                    # Render word by word
                    x_cursor = line_x
                    words_in_line = text_line.split()
                    ch_h = font_size
                    for wi, word in enumerate(words_in_line):
                        if x_cursor > col_x + col_width - 20:
                            break
                        word_start_char = len(char_boxes)

                        # Measure and render each character
                        for ch in word:
                            try:
                                bbox = font.getbbox(ch)
                                ch_w = bbox[2] - bbox[0]
                                ch_h = bbox[3] - bbox[1]
                            except Exception:
                                ch_w, ch_h = font_size // 2, font_size

                            ch_w = max(ch_w, 4)
                            ch_h = max(ch_h, 8)

                            char_boxes.append({
                                "char": ch, "x": x_cursor, "y": y_cursor,
                                "w": ch_w, "h": ch_h,
                                "cx": x_cursor + ch_w / 2, "cy": y_cursor + ch_h / 2,
                                "angle": angle, "line_id": line_id, "block_id": block_id,
                            })
                            # Draw character
                            color = (self.rng.randint(0, 40), self.rng.randint(0, 40), self.rng.randint(0, 40))
                            draw.text((x_cursor, y_cursor), ch, fill=color, font=font)
                            x_cursor += ch_w

                        word_end_char = len(char_boxes)
                        if word_end_char > word_start_char:
                            word_indices.append((word_start_char, word_end_char))
                            word_box = {
                                "text": word,
                                "x": char_boxes[word_start_char]["x"],
                                "y": char_boxes[word_start_char]["y"],
                                "w": x_cursor - char_boxes[word_start_char]["x"],
                                "h": ch_h,
                                "line_id": line_id, "block_id": block_id,
                            }
                            word_list.append(word_box)

                        x_cursor += font_size // 3  # Word spacing

                    line_spacing = int(font_size * 1.5)
                    line_heights[line_id] = line_spacing
                    line_data.append({
                        "line_id": line_id, "block_id": block_id,
                        "y": y_cursor, "height": ch_h,
                        "text": text_line,
                    })
                    block_lines.append(line_id)
                    line_id += 1
                    y_cursor += line_spacing

                block_data.append({
                    "block_id": block_id, "lines": block_lines,
                })
                block_id += 1
                y_cursor += self.rng.randint(10, 30)  # Inter-block spacing

        # Convert to numpy
        img_np = np.array(img)

        # Apply augmentations
        if apply_augmentation and self.rng.random() < 0.5:
            img_np = self._augment(img_np)

        # Generate target maps
        stride = 2
        map_h, map_w = H // stride, W // stride
        region_map = render_region_map(char_boxes, map_h, map_w, stride)
        affinity_map = render_affinity_map(char_boxes, word_indices, map_h, map_w, stride)
        orient_map = render_orientation_map(char_boxes, map_h, map_w, 32, stride)
        scale_map = render_scale_map(char_boxes, line_heights, self.scale_bin_edges, map_h, map_w, stride)
        text_mask = render_text_mask(char_boxes, map_h, map_w, stride)

        # Full reading-order transcription
        full_text = "\n".join(ld["text"] for ld in line_data)

        return {
            "image": img_np,
            "words": word_list,
            "char_boxes": char_boxes,
            "word_indices": word_indices,
            "lines": line_data,
            "blocks": block_data,
            "transcription": full_text,
            "line_heights": line_heights,
            "targets": {
                "region": region_map,
                "affinity": affinity_map,
                "orientation": orient_map,
                "scale": scale_map,
                "text_mask": text_mask,
            },
        }

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Apply camera-like photometric augmentations."""
        # Random brightness/contrast
        alpha = self.np_rng.uniform(0.8, 1.2)
        beta = self.np_rng.randint(-20, 20)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Random Gaussian blur
        if self.np_rng.random() < 0.3:
            k = self.rng.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # JPEG compression artifact
        if self.np_rng.random() < 0.3:
            quality = self.np_rng.randint(40, 85)
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        # Gaussian noise
        if self.np_rng.random() < 0.2:
            noise = self.np_rng.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img

    def generate_dataset(self, output_dir: str, num_samples: int = 64,
                         split: str = "train") -> str:
        """
        Generate a dataset split and save to disk.
        Returns path to the manifest JSON file.
        """
        split_dir = os.path.join(output_dir, split)
        img_dir = os.path.join(split_dir, "images")
        anno_dir = os.path.join(split_dir, "annotations")
        target_dir = os.path.join(split_dir, "targets")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)

        manifest = []
        for i in range(num_samples):
            sample = self.generate_page(apply_augmentation=(split == "train"))
            sample_id = f"{split}_{i:06d}"

            # Save image
            img_path = os.path.join(img_dir, f"{sample_id}.png")
            cv2.imwrite(img_path, cv2.cvtColor(sample["image"], cv2.COLOR_RGB2BGR))

            # Save annotations
            anno = {
                "id": sample_id,
                "words": sample["words"],
                "lines": sample["lines"],
                "blocks": sample["blocks"],
                "transcription": sample["transcription"],
            }
            anno_path = os.path.join(anno_dir, f"{sample_id}.json")
            with open(anno_path, "w") as f:
                json.dump(anno, f, indent=2)

            # Save target maps as npz
            target_path = os.path.join(target_dir, f"{sample_id}.npz")
            np.savez_compressed(target_path, **sample["targets"])

            manifest.append({
                "id": sample_id,
                "image": os.path.relpath(img_path, output_dir),
                "annotation": os.path.relpath(anno_path, output_dir),
                "targets": os.path.relpath(target_path, output_dir),
            })

        manifest_path = os.path.join(output_dir, f"{split}_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({"split": split, "count": num_samples, "seed": self.rng.getstate()[1][0],
                        "samples": manifest}, f, indent=2)
        return manifest_path


def generate_word_crops(words: List[dict], image: np.ndarray,
                        target_h: int = 32) -> List[Tuple[np.ndarray, str]]:
    """Extract word crops from a page image for recognizer training."""
    crops = []
    h, w = image.shape[:2]
    for wd in words:
        x1 = max(0, int(wd["x"]) - 2)
        y1 = max(0, int(wd["y"]) - 2)
        x2 = min(w, int(wd["x"] + wd["w"]) + 2)
        y2 = min(h, int(wd["y"] + wd["h"]) + 2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        # Resize to target height preserving aspect ratio
        crop_h, crop_w = crop.shape[:2]
        if crop_h < 4 or crop_w < 4:
            continue
        new_w = max(4, int(crop_w * target_h / crop_h))
        crop = cv2.resize(crop, (new_w, target_h))
        crops.append((crop, wd["text"]))
    return crops
