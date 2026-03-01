from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightUNetDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, out_ch: int = 3) -> None:
        super().__init__()
        self.b3 = ConvBlock(hidden_dim * 2, hidden_dim)
        self.b2 = ConvBlock(hidden_dim * 2, hidden_dim)
        self.b1 = ConvBlock(hidden_dim * 2, hidden_dim)
        self.out = nn.Conv2d(hidden_dim, out_ch, 1)

    def forward(self, feats: List[torch.Tensor], out_hw=None) -> torch.Tensor:
        assert len(feats) == 4, f"Need 4 feature maps, got {len(feats)}"
        f1, f2, f3, f4 = feats  # shallow->deep

        x = f4
        x = F.interpolate(x, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.b3(torch.cat([x, f3], dim=1))

        x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.b2(torch.cat([x, f2], dim=1))

        x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.b1(torch.cat([x, f1], dim=1))
        x = self.out(x)

        if out_hw is not None:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return x
