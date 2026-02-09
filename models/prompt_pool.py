"""Prompt pool storing learnable prompts for modality/degradation pairs."""

from __future__ import annotations

import torch
import torch.nn as nn


class PromptPool(nn.Module):
    """Learnable prompt vectors for each (modality, degradation) pair."""

    def __init__(self, num_modalities: int, num_degradations: int, prompt_dim: int) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.num_degradations = num_degradations
        self.prompt_dim = prompt_dim
        self.prompts = nn.Parameter(
            torch.randn(num_modalities, num_degradations, prompt_dim) * 0.02
        )

    def forward(
        self,
        prompt_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return weighted prompt vector of shape [B, prompt_dim]."""
        b = prompt_weights.shape[0]
        weights = prompt_weights.view(b, -1)
        prompt_bank = self.prompts.view(-1, self.prompt_dim)
        return weights @ prompt_bank
