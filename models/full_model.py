"""Full model combining encoder, predictors, prompt pool, and decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from models.degradation_predictor import DegradationPredictor
from models.modality_predictor import ModalityPredictor
from models.prompt_pool import PromptPool
from models.restormer_decoder import RestormerDecoder
from models.restormer_encoder import RestormerEncoder


@dataclass
class ModelOutput:
    """Container for model outputs."""

    reconstructed: torch.Tensor
    modality_probs: torch.Tensor
    degradation_probs: torch.Tensor
    prompt_weights: torch.Tensor


class FullModel(nn.Module):
    """Restormer-style autoencoder with prompt-conditioned encoder."""

    def __init__(
        self,
        channels: list[int] | None = None,
        prompt_dim: int = 64,
        num_degradations: int = 3,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128]
        self.encoder = RestormerEncoder(channels, prompt_dim)
        self.decoder = RestormerDecoder(channels)
        self.modality_predictor = ModalityPredictor(channels[2])
        self.degradation_predictor = DegradationPredictor(channels[0], num_degradations)
        self.prompt_pool = PromptPool(2, num_degradations, prompt_dim)
        self.num_prompts = self.prompt_pool.num_modalities * self.prompt_pool.num_degradations

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # First pass without prompts to predict modality and degradation
        zero_prompt = torch.zeros(x.size(0), self.prompt_pool.prompt_dim, device=x.device)
        feat1, feat2, feat3 = self.encoder(x, zero_prompt)
        modality_probs = self.modality_predictor(feat3)
        degradation_probs = self.degradation_predictor(feat1, modality_probs)
        prompt_weights = modality_probs.unsqueeze(2) * degradation_probs.unsqueeze(1)
        prompt_weights = prompt_weights.view(x.size(0), -1)
        prompt = self.prompt_pool(prompt_weights)
        feat1, feat2, feat3 = self.encoder(x, prompt)
        reconstructed = self.decoder(feat1, feat2, feat3)
        return ModelOutput(
            reconstructed=reconstructed,
            modality_probs=modality_probs,
            degradation_probs=degradation_probs,
            prompt_weights=prompt_weights,
        )

    def export_state(self) -> Dict[str, torch.Tensor]:
        """Return model state for checkpointing."""
        return self.state_dict()
