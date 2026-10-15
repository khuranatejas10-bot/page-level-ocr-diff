"""
OCR-Diff: Conditional diffusion super-resolution for text images.
Paper 2 (IEEE IoTJ 2024): Two-stage framework with conditional U-Net,
linear attention, feature extractor, and residual learning.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class _SinusoidalTimeEmb(nn.Module):
    """Sinusoidal timestep embedding: tau_t in R^{2K}."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=1)


class _ResBlock(nn.Module):
    """Residual block with optional time-conditioning."""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int = 128):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class _LinearAttention(nn.Module):
    """Linear attention O(N) complexity (Paper 2, Section III-A)."""
    def __init__(self, ch: int):
        super().__init__()
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-2)
        ctx = torch.bmm(k, v.transpose(1, 2))  # (B, C, C)
        out = torch.bmm(ctx, q).reshape(B, C, H, W)
        return self.gamma * out + x


class LRFeatureExtractor(nn.Module):
    """
    Feature extractor: 5 residual blocks + skip (Paper 2, Fig. 2).
    Input: upsampled LR crop x_up (B, 3, H, W).
    Output: feature x_f (B, 3, H, W).
    """
    def __init__(self, mid_ch: int = 64):
        super().__init__()
        self.conv_in = nn.Conv2d(3, mid_ch, 3, padding=1)
        self.blocks = nn.ModuleList([_ResBlock(mid_ch, mid_ch, time_dim=1) for _ in range(5)])
        self.conv_out = nn.Conv2d(mid_ch, 3, 3, padding=1)

    def forward(self, x_up: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x_up)
        h0 = h
        dummy_t = torch.zeros(x_up.shape[0], 1, device=x_up.device)
        for blk in self.blocks:
            h = blk(h, dummy_t)
        return self.conv_out(h + h0)


class ConditionalUNet(nn.Module):
    """
    Conditional U-Net for noise prediction (Paper 2, Fig. 1 & 2).
    Input: concat(X_t, x_f) = 6 channels.
    Output: predicted noise (3 channels).
    Architecture: 10 residual blocks, linear attention at bottleneck, skip connections.
    """
    def __init__(self, base_ch: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_emb = nn.Sequential(
            _SinusoidalTimeEmb(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.inc = nn.Conv2d(6, base_ch, 3, padding=1)
        # Encoder
        self.d1 = _ResBlock(base_ch, base_ch * 2, time_dim)
        self.d2 = _ResBlock(base_ch * 2, base_ch * 4, time_dim)
        self.down = nn.Conv2d(base_ch * 4, base_ch * 8, 3, stride=2, padding=1)
        # Bottleneck with linear attention
        self.mid1 = _ResBlock(base_ch * 8, base_ch * 8, time_dim)
        self.attn = _LinearAttention(base_ch * 8)
        self.mid2 = _ResBlock(base_ch * 8, base_ch * 8, time_dim)
        # Decoder
        self.up = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1)
        self.u1 = _ResBlock(base_ch * 8, base_ch * 2, time_dim)  # concat with d2 skip
        self.u2 = _ResBlock(base_ch * 4, base_ch, time_dim)      # concat with d1 skip
        self.outc = nn.Conv2d(base_ch * 2, 3, 3, padding=1)      # concat with inc skip

    def forward(self, x_t: torch.Tensor, x_f: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        h0 = self.inc(torch.cat([x_t, x_f], dim=1))
        h1 = self.d1(h0, t_emb)
        h2 = self.d2(h1, t_emb)
        h = self.down(h2)
        h = self.mid1(h, t_emb)
        h = self.attn(h)
        h = self.mid2(h, t_emb)
        h = self.up(h)
        if h.shape[2:] != h2.shape[2:]:
            h = F.interpolate(h, size=h2.shape[2:], mode="bilinear", align_corners=False)
        h = self.u1(torch.cat([h, h2], dim=1), t_emb)
        h = self.u2(torch.cat([h, h1], dim=1), t_emb)
        return self.outc(torch.cat([h, h0], dim=1))


class DiffusionSchedule:
    """Cosine beta schedule (Nichol & Dhariwal) with forward/reverse diffusion."""
    def __init__(self, T: int = 1000, device: str = "cpu"):
        self.T = T
        s = 8e-3
        steps = torch.arange(T + 1, dtype=torch.float64)
        f_t = torch.cos(((steps / T + s) / (1 + s)) * (math.pi / 2)) ** 2
        alpha_bar = f_t / f_t[0]
        betas = []
        for i in range(1, T + 1):
            b = 1.0 - (alpha_bar[i] / alpha_bar[i - 1])
            betas.append(min(max(b.item(), 1e-5), 0.999))
        self.betas = torch.tensor(betas, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.posterior_var = self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise, noise

    @torch.no_grad()
    def p_sample_ddpm(self, model_fn, x_t: torch.Tensor, t_val: int,
                      x_f: torch.Tensor) -> torch.Tensor:
        """Single DDPM reverse step."""
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t_val, device=x_t.device, dtype=torch.long)
        pred_noise = model_fn(x_t, x_f, t_tensor)
        alpha = self.alphas[t_val]
        alpha_bar = self.alpha_bar[t_val]
        coef = (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)
        mean = (1.0 / torch.sqrt(alpha)) * (x_t - coef * pred_noise)
        if t_val > 0:
            sigma = torch.sqrt(self.posterior_var[t_val])
            mean = mean + sigma * torch.randn_like(x_t)
        return mean

    @torch.no_grad()
    def ddim_sample(self, model_fn, x_f: torch.Tensor, shape: tuple,
                    num_steps: int = 20) -> torch.Tensor:
        """DDIM deterministic sampling for fast inference."""
        device = x_f.device
        x = torch.randn(shape, device=device)
        stride = max(1, self.T // num_steps)
        timesteps = list(range(0, self.T, stride))[::-1]
        for t_val in timesteps:
            B = x.shape[0]
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            pred_noise = model_fn(x, x_f, t)
            ab = self.alpha_bar[t_val]
            ab_prev = self.alpha_bar[max(0, t_val - stride)]
            x0_pred = (x - torch.sqrt(1 - ab) * pred_noise) / torch.sqrt(ab)
            x0_pred = x0_pred.clamp(-1, 1)
            x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1 - ab_prev) * pred_noise
        return x


class OCRDiffSR(nn.Module):
    """
    Complete OCR-Diff super-resolution module.
    Wraps feature extractor + conditional U-Net + diffusion schedule.
    Residual learning: X_hat = X_up + predicted_residual.
    """
    def __init__(self, base_ch: int = 64, time_dim: int = 128, T: int = 1000):
        super().__init__()
        self.feature_extractor = LRFeatureExtractor(mid_ch=base_ch)
        self.unet = ConditionalUNet(base_ch=base_ch, time_dim=time_dim)
        self.T = T
        self.schedule = None  # Created on correct device in .to()

    def init_schedule(self, device: str = "cpu"):
        self.schedule = DiffusionSchedule(T=self.T, device=device)

    def forward(self, x_t: torch.Tensor, x_up: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """Forward: predict noise given noisy HR, upsampled LR, and timestep."""
        x_f = self.feature_extractor(x_up)
        return self.unet(x_t, x_f, t)

    @torch.no_grad()
    def enhance(self, x_lr: torch.Tensor, hr_size: Tuple[int, int],
                num_steps: int = 20) -> torch.Tensor:
        """
        Enhance a low-resolution text crop.
        Args:
            x_lr: (B, 3, h, w) LR input in [0, 1]
            hr_size: (H, W) target HR size
            num_steps: DDIM sampling steps
        Returns:
            (B, 3, H, W) enhanced image in [0, 1]
        """
        x_up = F.interpolate(x_lr, size=hr_size, mode="bilinear", align_corners=False)
        x_f = self.feature_extractor(x_up)
        if self.schedule is None:
            self.init_schedule(x_lr.device.type)
        shape = (x_lr.shape[0], 3, hr_size[0], hr_size[1])
        x_hat_0 = self.schedule.ddim_sample(self.unet, x_f, shape, num_steps)
        # Residual learning: X_hat = X_up + X_hat_0
        return (x_up + x_hat_0).clamp(0, 1)
