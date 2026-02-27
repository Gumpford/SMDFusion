"""Prompt-driven gated modulation for multi-scale features."""

from __future__ import annotations

import torch
import torch.nn as nn


class _ScaleMapper(nn.Module):
    def __init__(self, prompt_dim: int, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(prompt_dim, channels)
        self.beta = nn.Linear(prompt_dim, channels)
        self.gate = nn.Linear(prompt_dim, channels)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, prompt_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = self.gamma(prompt_vec)
        beta = self.beta(prompt_vec)
        gate = torch.sigmoid(self.gate(prompt_vec))
        return gamma, beta, gate


class GatedModulation(nn.Module):
    """Apply prompt-conditioned affine modulation with learnable gates."""

    def __init__(self, prompt_dim: int, channels: list[int]) -> None:
        super().__init__()
        self.prompt_dim = prompt_dim
        self.channels = channels
        self.mappers = nn.ModuleList([_ScaleMapper(prompt_dim, c) for c in channels])

    def forward(self, feats: list[torch.Tensor], prompt_vec: torch.Tensor) -> list[torch.Tensor]:
        if len(feats) != len(self.channels):
            raise ValueError(f"Expected {len(self.channels)} feature maps, got {len(feats)}")
        if prompt_vec.ndim != 2 or prompt_vec.size(1) != self.prompt_dim:
            raise ValueError(f"Expected prompt_vec shape (B,{self.prompt_dim}), got {tuple(prompt_vec.shape)}")

        outs = []
        for i, feat in enumerate(feats):
            b, c, _, _ = feat.shape
            gamma, beta, gate = self.mappers[i](prompt_vec)
            gamma = gamma.view(b, c, 1, 1)
            beta = beta.view(b, c, 1, 1)
            gate = gate.view(b, c, 1, 1)
            mod = feat * (1.0 + gate * gamma) + gate * beta
            outs.append(mod)
        return outs
