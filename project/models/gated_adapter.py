from typing import Tuple

import torch
import torch.nn as nn


class GatedAdapter(nn.Module):
    """F' = F + alpha * (sigmoid(Wz) * T(F))."""

    def __init__(self, dim: int = 256, use_dwconv: bool = True) -> None:
        super().__init__()
        self.gate_fc = nn.Linear(dim, dim)
        layers = [nn.Conv2d(dim, dim, kernel_size=1), nn.GELU()]
        if use_dwconv:
            layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim), nn.GELU()]
        layers += [nn.Conv2d(dim, dim, kernel_size=1)]
        self.transform = nn.Sequential(*layers)
        self.alpha = nn.Parameter(torch.tensor(1e-3))

    def forward(self, feat: torch.Tensor, z: torch.Tensor, enable_prompt: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, _, _ = feat.shape
        assert z.shape == (b, c), f"Expected z shape {(b, c)}, got {tuple(z.shape)}"
        g = torch.sigmoid(self.gate_fc(z)).view(b, c, 1, 1)
        if not enable_prompt:
            return feat, g
        t = self.transform(feat)
        out = feat + self.alpha * (g * t)
        return out, g
