"""
Four-head text detector based on CRAFT architecture.
Paper 1 (IEEE Access 2025): Region, Affinity, Orientation (32-class), Scale (10-class).
Uses a ResNet backbone with FPN/U-Net decoder for pixel-level predictions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Tuple, Optional


class _DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _UpBlock(nn.Module):
    """Upsample + skip-concat + double-conv."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatches from non-power-of-2 inputs
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FourHeadDetector(nn.Module):
    """
    CRAFT-inspired detector producing 4 dense prediction maps:
      1. Region heatmap (1 ch, sigmoid) – character/word centers
      2. Affinity heatmap (1 ch, sigmoid) – inter-character links
      3. Orientation (num_orient ch, softmax) – text orientation class per pixel
      4. Scale (num_scale ch, softmax) – interline distance class per pixel

    Backbone: ResNet-18 or ResNet-34 with ImageNet init.
    Decoder: U-Net style with skip connections.
    Output stride: 2 (half the input resolution).
    """
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 num_orient: int = 32, num_scale: int = 10):
        super().__init__()
        self.num_orient = num_orient
        self.num_scale = num_scale

        # --- Encoder (ResNet backbone) ---
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif backbone == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        else:
            base = models.resnet18(weights=None)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)  # /2, 64ch
        self.pool = base.maxpool                                     # /4
        self.enc1 = base.layer1  # /4, 64ch
        self.enc2 = base.layer2  # /8, 128ch
        self.enc3 = base.layer3  # /16, 256ch
        self.enc4 = base.layer4  # /32, 512ch

        # --- Decoder (U-Net upsampling) ---
        self.up4 = _UpBlock(512, 256, 256)   # /16
        self.up3 = _UpBlock(256, 128, 128)   # /8
        self.up2 = _UpBlock(128, 64, 64)     # /4
        self.up1 = _UpBlock(64, 64, 64)      # /2 (output stride = 2)

        # --- Prediction heads (at stride=2) ---
        self.head_region = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1), nn.Sigmoid()
        )
        self.head_affinity = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1), nn.Sigmoid()
        )
        self.head_orientation = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_orient, 1)  # logits; cross-entropy applied externally
        )
        self.head_scale = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_scale, 1)  # logits
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input image, H and W should be divisible by 32.
        Returns:
            dict with keys 'region', 'affinity', 'orientation', 'scale'.
            Spatial dimensions are (H/2, W/2).
        """
        s0 = self.stem(x)     # (B, 64, H/2, W/2)
        s1 = self.enc1(self.pool(s0))  # (B, 64, H/4, W/4)
        s2 = self.enc2(s1)    # (B, 128, H/8, W/8)
        s3 = self.enc3(s2)    # (B, 256, H/16, W/16)
        s4 = self.enc4(s3)    # (B, 512, H/32, W/32)

        d3 = self.up4(s4, s3)  # (B, 256, H/16, W/16)
        d2 = self.up3(d3, s2)  # (B, 128, H/8, W/8)
        d1 = self.up2(d2, s1)  # (B, 64, H/4, W/4)
        d0 = self.up1(d1, s0)  # (B, 64, H/2, W/2)

        return {
            "region": self.head_region(d0),          # (B, 1, H/2, W/2)
            "affinity": self.head_affinity(d0),      # (B, 1, H/2, W/2)
            "orientation": self.head_orientation(d0), # (B, 32, H/2, W/2)
            "scale": self.head_scale(d0),            # (B, 10, H/2, W/2)
        }


class DetectorLoss(nn.Module):
    """
    Combined detection loss (Paper 1 Eq. 1-3):
    L = L_region + L_affinity + lambda_s * L_scale + lambda_o * L_orientation

    Region/Affinity: balanced MSE with hard-negative mining (CRAFT-style OHEM).
    Orientation/Scale: cross-entropy masked to text-positive pixels.
    """
    def __init__(self, lambda_scale: float = 0.1, lambda_orient: float = 0.3,
                 ohem_ratio: float = 3.0):
        super().__init__()
        self.lambda_scale = lambda_scale
        self.lambda_orient = lambda_orient
        self.ohem_ratio = ohem_ratio

    def _ohem_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Online hard-example mining MSE for region/affinity maps."""
        diff_sq = (pred - target) ** 2  # (B, 1, H, W)
        positive_mask = (target > 0.5).float()
        n_pos = max(positive_mask.sum().item(), 1.0)
        n_neg = min(n_pos * self.ohem_ratio, (1.0 - positive_mask).sum().item())

        pos_loss = (diff_sq * positive_mask).sum() / n_pos

        neg_losses = diff_sq * (1.0 - positive_mask)
        if n_neg > 0:
            neg_flat = neg_losses.reshape(-1)
            # Select top-k hardest negatives
            k = int(n_neg)
            if k > 0 and k <= neg_flat.numel():
                topk_vals, _ = torch.topk(neg_flat, k)
                neg_loss = topk_vals.mean()
            else:
                neg_loss = neg_flat.mean()
        else:
            neg_loss = torch.tensor(0.0, device=pred.device)

        return pos_loss + neg_loss

    def forward(self, preds: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            preds: detector output dict
            targets: dict with keys 'region_gt', 'affinity_gt' (B,1,H,W float),
                     'orientation_gt', 'scale_gt' (B,H,W long class indices),
                     'text_mask' (B,H,W float, 1 where text pixel).
        """
        loss_r = self._ohem_mse(preds["region"], targets["region_gt"])
        loss_a = self._ohem_mse(preds["affinity"], targets["affinity_gt"])

        text_mask = targets["text_mask"]  # (B, H, W)
        mask_sum = text_mask.sum().clamp(min=1.0)

        # Orientation CE masked to text pixels
        orient_logits = preds["orientation"]  # (B, 32, H, W)
        orient_gt = targets["orientation_gt"]  # (B, H, W) long
        ce_orient = F.cross_entropy(orient_logits, orient_gt, reduction="none")  # (B, H, W)
        loss_o = (ce_orient * text_mask).sum() / mask_sum

        # Scale CE masked to text pixels
        scale_logits = preds["scale"]  # (B, 10, H, W)
        scale_gt = targets["scale_gt"]  # (B, H, W) long
        ce_scale = F.cross_entropy(scale_logits, scale_gt, reduction="none")
        loss_s = (ce_scale * text_mask).sum() / mask_sum

        total = loss_r + loss_a + self.lambda_scale * loss_s + self.lambda_orient * loss_o

        return {
            "total": total, "region": loss_r, "affinity": loss_a,
            "orientation": loss_o, "scale": loss_s,
        }
