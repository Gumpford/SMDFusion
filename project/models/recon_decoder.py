import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ReconDecoder(nn.Module):
    def __init__(self, d=256, out_ch=3):
        super().__init__()
        self.b43 = ConvBlock(d * 2, d)
        self.b32 = ConvBlock(d * 2, d)
        self.b21 = ConvBlock(d * 2, d)
        self.out = nn.Conv2d(d, out_ch, 3, padding=1)

    def forward(self, feats):
        f1, f2, f3, f4 = feats
        x = f4
        x = F.interpolate(x, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.b43(torch.cat([x, f3], dim=1))
        x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.b32(torch.cat([x, f2], dim=1))
        x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.b21(torch.cat([x, f1], dim=1))
        x = self.out(x)
        return F.interpolate(x, scale_factor=14, mode="bilinear", align_corners=False)
