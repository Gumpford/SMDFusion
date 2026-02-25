"""Prompt bank utilities for hierarchical gating."""

from __future__ import annotations

import torch
import torch.nn as nn


class PromptBank(nn.Module):
    """Simple text/visual prompt banks with gating-based weighting."""

    def __init__(self, num_prompts: int, prompt_dim: int) -> None:
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.text_bank = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.02)
        self.visual_bank = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.02)

    def init_text_from_clip(self, *_args, **_kwargs) -> None:
        """Placeholder hook for future CLIP-based text prompt initialization."""

    def init_visual_from_clip(self, *_args, **_kwargs) -> None:
        """Placeholder hook for future CLIP-based visual prompt initialization."""

    def weighted_text(self, p: torch.Tensor) -> torch.Tensor:
        if p.ndim != 2 or p.size(1) != self.num_prompts:
            raise ValueError(f"Expected P shape (B, {self.num_prompts}), got {tuple(p.shape)}")
        out = p @ self.text_bank
        assert out.shape == (p.size(0), self.prompt_dim)
        return out

    def weighted_visual(self, p: torch.Tensor) -> torch.Tensor:
        if p.ndim != 2 or p.size(1) != self.num_prompts:
            raise ValueError(f"Expected P shape (B, {self.num_prompts}), got {tuple(p.shape)}")
        out = p @ self.visual_bank
        assert out.shape == (p.size(0), self.prompt_dim)
        return out


def save_predictor_promptbank_checkpoint(
    path: str,
    predictor: nn.Module,
    prompt_bank: PromptBank | None = None,
    extra: dict | None = None,
) -> None:
    payload: dict[str, object] = {"predictor": predictor.state_dict()}
    if prompt_bank is not None:
        payload["prompt_bank"] = prompt_bank.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_predictor_promptbank_checkpoint(
    path: str,
    predictor: nn.Module,
    prompt_bank: PromptBank | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    payload = torch.load(path, map_location=map_location)
    if "predictor" in payload:
        predictor.load_state_dict(payload["predictor"])
    else:
        predictor.load_state_dict(payload)
    if prompt_bank is not None and "prompt_bank" in payload:
        prompt_bank.load_state_dict(payload["prompt_bank"])
    return payload
