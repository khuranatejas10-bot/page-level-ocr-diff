"""Central configuration management for PageOCRDiff."""
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class DetectorConfig:
    """Detector (4-head) configuration."""
    backbone: str = "resnet18"
    pretrained_backbone: bool = True
    output_stride: int = 2
    num_orientation_bins: int = 32  # Paper 1: 32 bins over 180 degrees
    num_scale_bins: int = 10       # Paper 1: 10 scale levels
    # Scale bin boundaries in pixels (interline distance); paper does not specify exact
    # boundaries, so we use a geometric progression from ~8px to ~80px.
    scale_bin_edges: List[float] = field(default_factory=lambda: [
        0.0, 8.0, 12.0, 18.0, 26.0, 36.0, 48.0, 62.0, 78.0, 96.0, 1e6
    ])
    # Loss weights (Paper 1 reference: lambda_s=0.1, lambda_o=0.3)
    lambda_scale: float = 0.1
    lambda_orientation: float = 0.3
    # Training
    lr: float = 1e-4
    lr_decay_factor: float = 0.8
    lr_decay_every: int = 2000
    max_iterations: int = 50000
    batch_size: int = 4
    image_size: int = 768  # Paper reference: 768x768
    # Component extraction
    region_threshold: float = 0.4
    affinity_threshold: float = 0.4
    min_component_area: int = 10
    min_component_score: float = 0.3
    # Block grouping (Paper 1: epsilon=2)
    delaunay_epsilon: float = 2.0
    # Line grouping
    max_poly_degree: int = 4
    line_deviation_scale_factor: float = 0.25  # Paper: s_bar / 4


@dataclass
class RecognizerConfig:
    """Word/line recognizer configuration."""
    backbone: str = "resnet_small"
    use_bilstm: bool = False
    bilstm_hidden: int = 256
    img_height: int = 32
    img_width: int = 128  # Max width; variable-width batching with padding
    # Vocabulary: printable ASCII minus some rare symbols
    charset: str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
    blank_index: int = 0  # CTC blank
    lr: float = 1e-4
    batch_size: int = 64
    max_iterations: int = 30000
    grad_clip: float = 5.0


@dataclass
class DiffusionSRConfig:
    """OCR-Diff diffusion super-resolution configuration."""
    # Resource-aware defaults (smaller than paper reference)
    T: int = 1000           # Paper reference: 3000; resource-aware: 1000
    lr_pretrain: float = 1e-4
    lr_finetune: float = 5e-5
    hr_height: int = 32     # Paper reference: 32
    hr_width: int = 128     # Paper reference: 128
    lr_height: int = 16     # Paper reference: 16
    lr_width: int = 64      # Paper reference: 64
    # U-Net
    base_channels: int = 64
    num_res_blocks: int = 10
    time_emb_dim: int = 128
    # Feature extractor
    fe_channels: int = 64
    fe_num_blocks: int = 5
    # Inference
    ddim_steps: int = 20    # Paper used full T; resource-aware: DDIM 20 steps
    use_ddim: bool = True
    # Fine-tuning
    recon_weight: float = 1.0
    recog_weight: float = 0.01
    # Confidence gate
    enhancement_confidence_threshold: float = 0.5
    batch_size: int = 16
    max_iterations: int = 20000


@dataclass
class Config:
    """Top-level project configuration."""
    project_name: str = "PageOCRDiff"
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
    data_root: str = "data"
    output_dir: str = "runs"
    num_workers: int = 0  # 0 for Windows compatibility
    mixed_precision: bool = False
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    recognizer: RecognizerConfig = field(default_factory=RecognizerConfig)
    diffusion_sr: DiffusionSRConfig = field(default_factory=DiffusionSRConfig)

    def resolve_device(self):
        """Resolve 'auto' device to actual device."""
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file, merging with defaults."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls()
        _merge_dict_into_dataclass(cfg, data)
        return cfg

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        import dataclasses
        d = dataclasses.asdict(self)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)


def _merge_dict_into_dataclass(obj, d: dict):
    """Recursively merge dict values into a dataclass instance."""
    import dataclasses
    for key, val in d.items():
        if hasattr(obj, key):
            attr = getattr(obj, key)
            if dataclasses.is_dataclass(attr) and isinstance(val, dict):
                _merge_dict_into_dataclass(attr, val)
            else:
                setattr(obj, key, val)
