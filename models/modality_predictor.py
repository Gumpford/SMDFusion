"""Modality predictor using low-resolution features."""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalityPredictor(nn.Module):
    """Predicts modality probabilities from low-resolution features."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.pool(feat))
        return torch.softmax(logits, dim=1)
