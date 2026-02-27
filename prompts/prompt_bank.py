"""Prompt bank modules for reconstruction with hierarchical gating."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptBank(nn.Module):
    """Learnable prompt pool bank of shape (K, D)."""

    def __init__(self, k: int, d: int) -> None:
        super().__init__()
        self.k = k
        self.d = d
        self.bank = nn.Parameter(torch.randn(k, d) * 0.02)

    def normalize_bank(self, mode: str = "none") -> None:
        """In-place normalization for stability."""
        with torch.no_grad():
            if mode == "none":
                return
            if mode == "l2":
                self.bank.copy_(F.normalize(self.bank, dim=-1))
                return
            raise ValueError(f"Unsupported normalize mode: {mode}")

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        if p.ndim != 2 or p.size(1) != self.k:
            raise ValueError(f"Expected P shape (B, {self.k}), got {tuple(p.shape)}")
        prompt = p @ self.bank
        assert prompt.shape == (p.size(0), self.d)
        return prompt


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
