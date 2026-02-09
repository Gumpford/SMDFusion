"""Prompt regularization utilities including IDF-inspired entropy weighting."""

from __future__ import annotations

import torch
import torch.nn as nn


class PromptRegularization(nn.Module):
    """Compute orthogonality loss across prompts."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, prompts: torch.Tensor) -> torch.Tensor:
        num_prompts, dim = prompts.shape
        if num_prompts <= 1:
            return torch.tensor(0.0, device=prompts.device)
        norms = prompts / (prompts.norm(dim=1, keepdim=True) + 1e-8)
        similarity = norms @ norms.t()
        identity = torch.eye(num_prompts, device=prompts.device)
        return ((similarity - identity) ** 2).sum() / (num_prompts * num_prompts)


class PromptFrequencyTracker:
    """Track prompt usage frequency with exponential moving average."""

    def __init__(self, num_prompts: int, momentum: float = 0.99) -> None:
        self.num_prompts = num_prompts
        self.momentum = momentum
        self.ema = torch.zeros(num_prompts)

    def update(self, prompt_weights: torch.Tensor) -> None:
        """Update EMA of prompt frequencies from batch prompt weights."""
        # Added for IDF prompt regularization
        batch_mean = prompt_weights.mean(dim=0)
        if self.ema.device != batch_mean.device:
            self.ema = self.ema.to(batch_mean.device)
        self.ema = self.momentum * self.ema + (1.0 - self.momentum) * batch_mean

    def get_idf_weight(self) -> torch.Tensor:
        """Return IDF-inspired weights on the same device as EMA."""
        # Added for IDF prompt regularization
        eps = 1e-8
        return torch.log(1.0 / (self.ema + eps))


def weighted_prompt_entropy(prompt_weights: torch.Tensor, idf_weight: torch.Tensor) -> torch.Tensor:
    """Return weighted entropy-like term sum_k idf_weight[k] * (p_k * log p_k)."""
    # Added for IDF prompt regularization
    eps = 1e-8
    weighted = prompt_weights * torch.log(prompt_weights + eps)
    return (weighted * idf_weight).sum(dim=1).mean()
