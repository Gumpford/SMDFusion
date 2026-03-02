from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


_DINO_OUT_DIM = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}


class FrozenDINOv2Encoder(nn.Module):
    """Shared frozen DINOv2 encoder with trainable 1x1 projections to hidden_dim."""

    def __init__(self, model_name: str = "dinov2_vitb14", hidden_dim: int = 256) -> None:
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", model_name)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False

        in_dim = _DINO_OUT_DIM.get(model_name, 768)
        self.proj = nn.ModuleList([nn.Conv2d(in_dim, hidden_dim, 1) for _ in range(4)])

        # DINOv2 commonly uses ImageNet style normalization.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _extract_tokens(self, x3: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            outs = self.dino.get_intermediate_layers(x3, n=4, reshape=False)
        return list(outs)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"Expected BCHW, got {tuple(x.shape)}")
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.size(1) != 3:
            raise ValueError(f"DINO input channels must be 1 or 3, got {x.size(1)}")

        x = (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)
        h, w = x.shape[-2:]
        h14, w14 = max(14, (h // 14) * 14), max(14, (w // 14) * 14)
        if h14 != h or w14 != w:
            x = F.interpolate(x, size=(h14, w14), mode="bilinear", align_corners=False)

        tokens = self._extract_tokens(x)
        feats: List[torch.Tensor] = []
        for i, t in enumerate(tokens):
            if t.dim() != 3:
                raise RuntimeError(f"Unexpected token shape: {tuple(t.shape)}")
            b, n, c = t.shape
            side = int(n ** 0.5)
            if side * side != n:
                raise RuntimeError(f"Token number {n} is not square.")
            fmap = t.transpose(1, 2).reshape(b, c, side, side)
            feats.append(self.proj[i](fmap))
        return feats
