import torch
import torch.nn as nn


class GatedAdapter(nn.Module):
    def __init__(self, d: int = 256):
        super().__init__()
        self.gate = nn.Linear(d, d)
        self.tfm = nn.Sequential(
            nn.Conv2d(d, d, 1),
            nn.GELU(),
            nn.Conv2d(d, d, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(1e-3))

    def forward(self, f: torch.Tensor, z: torch.Tensor, enable_prompt: bool = True):
        g = torch.sigmoid(self.gate(z)).unsqueeze(-1).unsqueeze(-1)
        if enable_prompt:
            out = f + self.alpha * (g * self.tfm(f))
        else:
            out = f
        return out, g
