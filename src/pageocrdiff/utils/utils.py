"""General utilities: seeding, checkpointing, logging, visualisation."""
import os
import json
import random
import time
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional


def set_seed(seed: int = 42):
    """Set all global random seeds deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a clean console + file logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                datefmt="%H:%M:%S")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


class AverageMeter:
    """Tracks running mean of a scalar metric."""
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


class Checkpointer:
    """Saves and loads model/optimizer/step checkpoints."""
    def __init__(self, output_dir: str, max_keep: int = 3):
        self.output_dir = output_dir
        self.max_keep = max_keep
        os.makedirs(output_dir, exist_ok=True)
        self._saved: list = []

    def save(self, tag: str, model: torch.nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None,
             step: int = 0, metrics: Optional[Dict[str, float]] = None):
        path = os.path.join(self.output_dir, f"ckpt_{tag}.pt")
        state = {
            "step": step,
            "model": model.state_dict(),
            "metrics": metrics or {},
        }
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        torch.save(state, path)
        self._saved.append(path)
        # Evict old checkpoints
        while len(self._saved) > self.max_keep:
            old = self._saved.pop(0)
            if os.path.exists(old):
                os.remove(old)
        # Always save a "best" symlink
        best_path = os.path.join(self.output_dir, "ckpt_best.pt")
        if metrics:
            torch.save(state, best_path)
        return path

    @staticmethod
    def load(path: str, model: torch.nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None,
             device: str = "cpu") -> Dict[str, Any]:
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if optimizer and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        return state


def save_metrics(metrics: Dict[str, float], path: str):
    """Append a metrics record (with timestamp) to a JSONL file."""
    record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **metrics}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_manifest(manifest_path: str) -> list:
    """Load a dataset manifest JSON file."""
    with open(manifest_path, "r") as f:
        data = json.load(f)
    return data.get("samples", data) if isinstance(data, dict) else data


def image_to_tensor(image: np.ndarray,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    """Normalise a (H, W, 3) uint8 BGR image to an ImageNet-normalised tensor."""
    img = image[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


def visualise_detections(image: np.ndarray,
                         word_boxes: list,
                         output_path: Optional[str] = None) -> np.ndarray:
    """Draw word bounding boxes on image and optionally save."""
    import cv2
    vis = image.copy()
    for wb in word_boxes:
        x, y, w, h = int(wb["bbox"][0]), int(wb["bbox"][1]), int(wb["bbox"][2]), int(wb["bbox"][3])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 180, 0), 1)
        if wb.get("text"):
            cv2.putText(vis, wb["text"][:12], (x, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 200), 1)
    if output_path:
        cv2.imwrite(output_path, vis)
    return vis
