"""Reconstruction loss using L1 distance."""

from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """Compute L1 reconstruction loss."""

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(prediction, target)
