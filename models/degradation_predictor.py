"""Degradation predictor conditioned on modality probabilities."""

from __future__ import annotations

import torch
import torch.nn as nn


class DegradationPredictor(nn.Module):
    """Predicts degradation probabilities from high-resolution features."""

    def __init__(self, in_channels: int, num_degradations: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels + 2, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, num_degradations),
        )

    def forward(self, feat: torch.Tensor, modality_probs: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(feat).flatten(1)
        x = torch.cat([pooled, modality_probs], dim=1)
        logits = self.classifier(x)
        return torch.softmax(logits, dim=1)
