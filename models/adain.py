"""Adaptive Instance Normalization (AdaIN) for prompt injection."""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaIN(nn.Module):
    """Applies AdaIN using a style vector mapped to channel-wise mean and std."""

    def __init__(self, num_features: int, style_dim: int) -> None:
        super().__init__()
        self.to_mean = nn.Linear(style_dim, num_features)
        self.to_std = nn.Linear(style_dim, num_features)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Normalize features and modulate with style-derived statistics."""
        b, c, h, w = x.shape
        mean = x.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        std = x.view(b, c, -1).std(dim=2).view(b, c, 1, 1) + 1e-6
        style_mean = self.to_mean(style).view(b, c, 1, 1)
        style_std = torch.exp(self.to_std(style)).view(b, c, 1, 1)
        return (x - mean) / std * style_std + style_mean
