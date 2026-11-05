"""
PyTorch Dataset classes for detector and recognizer training.
Supports: local synthetic data (always available), HierText, SynthText adapters.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Dict
import cv2


class DetectorDataset(Dataset):
    """
    Dataset for training the four-head detector.
    Reads from a manifest produced by SyntheticPageGenerator.generate_dataset().
    Returns image tensor + 4 target tensors.
    """
    def __init__(self, manifest_path: str, data_root: str,
                 img_size: int = 768, augment: bool = True):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.samples = manifest.get("samples", manifest)
        self.data_root = data_root
        self.img_size = img_size
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img_path = os.path.join(self.data_root, sample["image"])
        target_path = os.path.join(self.data_root, sample["targets"])

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 200
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to img_size
        h, w = img.shape[:2]
        if max(h, w) != self.img_size:
            scale = self.img_size / max(h, w)
            nw, nh = int(w * scale), int(h * scale)
            img = cv2.resize(img, (nw, nh))
        # Pad to img_size x img_size
        ph = img.shape[0]; pw = img.shape[1]
        pad_h = max(0, self.img_size - ph)
        pad_w = max(0, self.img_size - pw)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)),
                     mode="constant", constant_values=255)[:self.img_size, :self.img_size]

        # Normalise
        img_f = img.astype(np.float32) / 255.0
        img_f = (img_f - self.mean) / self.std

        # Load targets
        targets = np.load(target_path)
        map_size = self.img_size // 2

        def _resize_map(arr, out_h, out_w, interp=cv2.INTER_LINEAR):
            return cv2.resize(arr.astype(np.float32), (out_w, out_h), interpolation=interp)

        region = _resize_map(targets["region"], map_size, map_size)
        affinity = _resize_map(targets["affinity"], map_size, map_size)
        text_mask = _resize_map(targets["text_mask"], map_size, map_size)
        orient = cv2.resize(targets["orientation"].astype(np.float32),
                            (map_size, map_size), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        scale_t = cv2.resize(targets["scale"].astype(np.float32),
                             (map_size, map_size), interpolation=cv2.INTER_NEAREST).astype(np.int64)

        img_tensor = torch.from_numpy(img_f.transpose(2, 0, 1)).float()

        return {
            "image": img_tensor,
            "region_gt": torch.from_numpy(region).unsqueeze(0).float(),
            "affinity_gt": torch.from_numpy(affinity).unsqueeze(0).float(),
            "orientation_gt": torch.from_numpy(orient).long(),
            "scale_gt": torch.from_numpy(scale_t).long(),
            "text_mask": torch.from_numpy(text_mask).float(),
        }


class RecognizerDataset(Dataset):
    """
    Dataset for training the CTC recognizer.
    Reads word crops (image path + text) from a JSONL file.
    """
    def __init__(self, samples: List[Tuple[np.ndarray, str]],
                 vocab, img_height: int = 32, img_width: int = 128):
        self.samples = samples
        self.vocab = vocab
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        crop, text = self.samples[idx]
        # Resize to fixed height, preserve aspect
        h, w = crop.shape[:2]
        if h < 2:
            h = 2
        new_w = max(4, int(w * self.img_height / h))
        crop_r = cv2.resize(crop, (new_w, self.img_height))
        if crop_r.ndim == 3:
            crop_r = cv2.cvtColor(crop_r, cv2.COLOR_BGR2GRAY)
        # Pad/truncate to img_width
        cw = min(crop_r.shape[1], self.img_width)
        padded = np.full((self.img_height, self.img_width), 128, dtype=np.uint8)
        padded[:, :cw] = crop_r[:, :cw]
        img_f = (padded.astype(np.float32) / 127.5 - 1.0)

        label = self.vocab.encode(text)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "image": torch.from_numpy(img_f).unsqueeze(0).float(),
            "label": label_tensor,
            "label_len": torch.tensor(len(label), dtype=torch.long),
        }


def recognizer_collate(batch):
    """Custom collate for variable-length CTC labels."""
    images = torch.stack([b["image"] for b in batch])
    labels = torch.cat([b["label"] for b in batch])
    label_lens = torch.stack([b["label_len"] for b in batch])
    return {"image": images, "label": labels, "label_len": label_lens}


class SRPairDataset(Dataset):
    """
    LR/HR paired dataset for diffusion SR training.
    Synthesises pairs by downsampling word crops on the fly.
    """
    def __init__(self, word_crops: List[Tuple[np.ndarray, str]],
                 hr_size: Tuple[int, int] = (32, 128),
                 lr_size: Tuple[int, int] = (16, 64)):
        self.crops = word_crops
        self.hr_h, self.hr_w = hr_size
        self.lr_h, self.lr_w = lr_size

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        crop, _ = self.crops[idx]
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        hr = cv2.resize(crop, (self.hr_w, self.hr_h))
        lr = cv2.resize(hr, (self.lr_w, self.lr_h))

        hr_t = torch.from_numpy(hr[:, :, ::-1].copy().transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0)
        lr_t = torch.from_numpy(lr[:, :, ::-1].copy().transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0)

        # Upsampled LR (input to diffusion)
        lr_up = torch.nn.functional.interpolate(lr_t.unsqueeze(0),
                                                size=(self.hr_h, self.hr_w),
                                                mode="bilinear", align_corners=False).squeeze(0)
        return {"hr": hr_t, "lr": lr_t, "lr_up": lr_up}
