"""Trainer implementation for self-supervised Restormer prompt model."""

from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from losses.entropy_loss import EntropyLoss
from losses.prompt_regularization import (
    PromptFrequencyTracker,
    PromptRegularization,
    weighted_prompt_entropy,
)
from losses.reconstruction_loss import ReconstructionLoss
from models.full_model import FullModel


class Trainer:
    """Training helper for model optimization and checkpointing."""

    def __init__(
        self,
        model: FullModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        lambda_prompt: float = 0.1,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda_prompt = lambda_prompt
        self.recon_loss = ReconstructionLoss()
        self.entropy_loss = EntropyLoss()
        self.prompt_reg = PromptRegularization()
        self.prompt_tracker = PromptFrequencyTracker(num_prompts=model.num_prompts)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        inputs = batch["input"].to(self.device)
        targets = batch["target"].to(self.device)
        output = self.model(inputs)
        loss_recon = self.recon_loss(output.reconstructed, targets)
        loss_modality_entropy = self.entropy_loss(output.modality_probs)
        loss_degradation_entropy = self.entropy_loss(output.degradation_probs)
        prompts = self.model.prompt_pool.prompts.view(-1, self.model.prompt_pool.prompt_dim)
        loss_prompt = self.prompt_reg(prompts)
        # Added for IDF prompt regularization
        self.prompt_tracker.update(output.prompt_weights.detach())
        idf_weight = self.prompt_tracker.get_idf_weight()
        loss_prompt_entropy = weighted_prompt_entropy(output.prompt_weights, idf_weight)
        loss = (
            loss_recon
            + 0.1 * (loss_modality_entropy + loss_degradation_entropy)
            + 0.1 * loss_prompt
            + self.lambda_prompt * loss_prompt_entropy
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "total": loss.item(),
            "reconstruction": loss_recon.item(),
            "entropy_modality": loss_modality_entropy.item(),
            "entropy_degradation": loss_degradation_entropy.item(),
            "prompt_reg": loss_prompt.item(),
            "prompt_entropy": loss_prompt_entropy.item(),
        }

    def fit(self, dataloader: DataLoader, epochs: int) -> None:
        for epoch in range(1, epochs + 1):
            for step, batch in enumerate(dataloader, start=1):
                losses = self.train_step(batch)
                print(
                    f"Epoch {epoch} Step {step} | "
                    + " ".join([f"{key}: {value:.4f}" for key, value in losses.items()])
                )

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
