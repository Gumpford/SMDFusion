"""Focal loss implementations."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassFocalLoss(nn.Module):
    """Multi-class focal loss taking logits as input."""

    def __init__(
        self,
        alpha: Sequence[float] | torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            alpha_tensor = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha_tensor.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"Expected logits shape (B, C), got {tuple(logits.shape)}")
        if target.ndim != 1:
            raise ValueError(f"Expected target shape (B,), got {tuple(target.shape)}")
        if logits.size(0) != target.size(0):
            raise ValueError("Batch size mismatch between logits and target")

        log_probs = F.log_softmax(logits, dim=-1)
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return logits.new_zeros(()) if self.reduction != "none" else logits.new_zeros(target.shape)

        safe_target = target.clone()
        safe_target[~valid_mask] = 0
        log_pt = log_probs.gather(dim=-1, index=safe_target.unsqueeze(-1)).squeeze(-1)
        pt = log_pt.exp()

        if self.alpha is not None:
            if self.alpha.numel() != logits.size(1):
                raise ValueError("Alpha length must match number of classes")
            alpha_t = self.alpha.to(logits.device)[safe_target]
        else:
            alpha_t = torch.ones_like(pt)

        loss = -alpha_t * ((1.0 - pt) ** self.gamma) * log_pt
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss[valid_mask].mean()
