from __future__ import annotations

import random

import torch
import torch.nn.functional as F


IR_DEG_NAMES = [
    "0_noise",
    "1_blur",
    "2_contrast",
    "3_stripe_fpn",
    "4_dead_pixels",
    "5_gain_offset",
]

VI_DEG_NAMES = [
    "0_lowlight",
    "1_overexposure",
    "2_jpeg",
    "3_motion_blur",
    "4_haze",
    "5_rain_snow",
]


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def _gaussian_blur(img: torch.Tensor, k: int, sigma: float) -> torch.Tensor:
    radius = k // 2
    ax = torch.arange(-radius, radius + 1, device=img.device, dtype=img.dtype)
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel2d = (g[:, None] * g[None, :]).expand(img.shape[1], 1, k, k)
    return F.conv2d(img, kernel2d, padding=radius, groups=img.shape[1])


def _motion_blur(img: torch.Tensor, k: int) -> torch.Tensor:
    kernel = torch.zeros((img.shape[1], 1, k, k), device=img.device, dtype=img.dtype)
    kernel[:, :, k // 2, :] = 1.0 / k
    return F.conv2d(img, kernel, padding=k // 2, groups=img.shape[1])


def _jpeg_like(img: torch.Tensor, block: int = 8, quality: int = 30) -> torch.Tensor:
    q = max(5, min(95, quality))
    scale = max(1, 100 // q)
    h, w = img.shape[-2:]
    h2 = max(block, (h // block // scale) * block)
    w2 = max(block, (w // block // scale) * block)
    down = F.interpolate(img, size=(h2, w2), mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(h, w), mode="nearest")
    return up


def _haze(img: torch.Tensor, amount: float) -> torch.Tensor:
    airlight = torch.rand((img.shape[0], 3, 1, 1), device=img.device, dtype=img.dtype) * 0.3 + 0.7
    return img * (1 - amount) + airlight * amount


def _rain_snow(img: torch.Tensor, density: float) -> torch.Tensor:
    b, c, h, w = img.shape
    noise = torch.rand((b, 1, h, w), device=img.device, dtype=img.dtype)
    mask = (noise < density).float()
    streak = F.avg_pool2d(mask, kernel_size=(7, 1), stride=1, padding=(3, 0))
    streak = torch.clamp(streak * 3.0, 0.0, 1.0)
    return torch.clamp(img + streak, 0.0, 1.0)


class OnlineSingleDegrader:
    def __init__(self) -> None:
        self.ir_names = IR_DEG_NAMES
        self.vi_names = VI_DEG_NAMES

    def degrade(self, clean: torch.Tensor, modality: str) -> tuple[torch.Tensor, int]:
        """Apply one random degradation according to modality.

        Args:
            clean: (B,3,H,W) in [0,1]
            modality: "ir" or "vi"
        Returns:
            degraded: (B,3,H,W), deg_id in [0..5]
        """
        deg_id = random.randint(0, 5)
        if modality.lower() == "ir":
            out = self._degrade_ir(clean, deg_id)
        elif modality.lower() == "vi":
            out = self._degrade_vi(clean, deg_id)
        else:
            raise ValueError(f"Unknown modality: {modality}")
        return _clamp01(out), deg_id

    def _degrade_ir(self, x: torch.Tensor, deg_id: int) -> torch.Tensor:
        if deg_id == 0:  # noise
            std = random.uniform(0.01, 0.08)
            return x + torch.randn_like(x) * std
        if deg_id == 1:  # blur
            k = random.choice([3, 5, 7])
            sigma = random.uniform(0.6, 2.0)
            return _gaussian_blur(x, k=k, sigma=sigma)
        if deg_id == 2:  # contrast
            alpha = random.uniform(0.5, 1.6)
            mean = x.mean(dim=(-2, -1), keepdim=True)
            return (x - mean) * alpha + mean
        if deg_id == 3:  # stripe_fpn
            b, c, h, w = x.shape
            col_noise = torch.randn((b, 1, 1, w), device=x.device, dtype=x.dtype) * random.uniform(0.02, 0.12)
            stripe = col_noise.expand(b, c, h, w)
            return x + stripe
        if deg_id == 4:  # dead_pixels
            p = random.uniform(0.001, 0.02)
            mask = (torch.rand_like(x[:, :1]) < p).float()
            dead_value = random.choice([0.0, 1.0])
            return x * (1 - mask) + dead_value * mask
        if deg_id == 5:  # gain_offset
            gain = random.uniform(0.7, 1.3)
            offset = random.uniform(-0.12, 0.12)
            return x * gain + offset
        raise ValueError(f"Invalid IR deg id {deg_id}")

    def _degrade_vi(self, x: torch.Tensor, deg_id: int) -> torch.Tensor:
        if deg_id == 0:  # lowlight
            gamma = random.uniform(1.5, 3.0)
            return torch.pow(torch.clamp(x, 1e-6, 1.0), gamma)
        if deg_id == 1:  # overexposure
            factor = random.uniform(1.1, 1.8)
            bias = random.uniform(0.03, 0.15)
            return x * factor + bias
        if deg_id == 2:  # jpeg
            quality = random.randint(10, 45)
            return _jpeg_like(x, block=8, quality=quality)
        if deg_id == 3:  # motion_blur
            k = random.choice([7, 9, 11])
            return _motion_blur(x, k)
        if deg_id == 4:  # haze
            amount = random.uniform(0.08, 0.35)
            return _haze(x, amount)
        if deg_id == 5:  # rain_snow
            density = random.uniform(0.01, 0.06)
            return _rain_snow(x, density)
        raise ValueError(f"Invalid VI deg id {deg_id}")
