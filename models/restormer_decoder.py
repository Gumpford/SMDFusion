"""Restormer-style decoder using lightweight transformer blocks."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from models.restormer_encoder import RestormerBlock


class RestormerDecoder(nn.Module):
    """Decoder that upsamples back to original resolution."""

    def __init__(self, channels: List[int]) -> None:
        super().__init__()
        self.stage3 = RestormerBlock(channels[2], num_heads=4)
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.stage2 = RestormerBlock(channels[1] * 2, num_heads=4)
        self.up1 = nn.ConvTranspose2d(channels[1] * 2, channels[0], kernel_size=4, stride=4)
        self.stage1 = RestormerBlock(channels[0] * 2, num_heads=2)
        self.output = nn.Conv2d(channels[0] * 2, 3, kernel_size=3, padding=1)

    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        feat3: torch.Tensor,
    ) -> torch.Tensor:
        x = self.stage3(feat3)
        x = self.up2(x)
        x = torch.cat([x, feat2], dim=1)
        x = self.stage2(x)
        x = self.up1(x)
        x = torch.cat([x, feat1], dim=1)
        x = self.stage1(x)
        return self.output(x)
