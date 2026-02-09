"""Entropy regularization for probability distributions."""

from __future__ import annotations

import torch
import torch.nn as nn


class EntropyLoss(nn.Module):
    """Encourage high-entropy (uncertain) predictions."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
        return entropy.mean()
