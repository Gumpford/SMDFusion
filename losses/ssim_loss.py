"""Lightweight SSIM implementation for grayscale tensors."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device: torch.device | None = None) -> torch.Tensor:
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    w = g[:, None] @ g[None, :]
    return w.unsqueeze(0).unsqueeze(0)


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.ndim != 4:
        raise ValueError("Expected input shape (B, C, H, W)")

    c = x.size(1)
    window = _gaussian_window(window_size, sigma, device=x.device).to(x.dtype)
    window = window.expand(c, 1, window_size, window_size)
    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=c)
    mu_y = F.conv2d(y, window, padding=pad, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=c) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.mean()


def ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim(x, y)
