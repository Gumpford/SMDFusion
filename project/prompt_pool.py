from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptPool(nn.Module):
    """12 learnable prompts composed by predictor probability P."""

    def __init__(self, num_prompts: int = 12, channels: int = 3, base_hw: tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        self.num_prompts = num_prompts
        self.channels = channels
        self.base_hw = base_hw
        self.prompt_bank = nn.Parameter(torch.empty(num_prompts, channels, *base_hw))
        nn.init.normal_(self.prompt_bank, mean=0.0, std=0.02)

    def compose(self, p: torch.Tensor, out_hw: tuple[int, int] | None = None) -> torch.Tensor:
        """Compose prompt from per-sample probability P.

        Args:
            p: (B, 12)
            out_hw: output spatial shape (H, W)
        Returns:
            (B, 3, H, W)
        """
        if p.ndim != 2 or p.shape[1] != self.num_prompts:
            raise ValueError(f"Expected p shape (B, {self.num_prompts}), got {tuple(p.shape)}")

        prompt = torch.einsum("bn,nchw->bchw", p, self.prompt_bank)
        if out_hw is not None and out_hw != self.base_hw:
            prompt = F.interpolate(prompt, size=out_hw, mode="bilinear", align_corners=False)
        return prompt

    def forward(self, p: torch.Tensor, out_hw: tuple[int, int] | None = None) -> torch.Tensor:
        return self.compose(p, out_hw=out_hw)
