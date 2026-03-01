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
    """Frozen DINOv2 encoder extracting 4 intermediate maps."""

    def __init__(self, model_name: str = "dinov2_vitb14", hidden_dim: int = 256, local_ckpt: str = "") -> None:
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        if local_ckpt:
            dino = torch.load(local_ckpt, map_location="cpu")
        else:
            dino = torch.hub.load("facebookresearch/dinov2", model_name)
        self.dino = dino
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()

        in_dim = _DINO_OUT_DIM.get(model_name, 768)
        self.proj = nn.ModuleList([nn.Conv2d(in_dim, hidden_dim, kernel_size=1) for _ in range(4)])

    @torch.no_grad()
    def _forward_tokens(self, x3: torch.Tensor) -> List[torch.Tensor]:
        if hasattr(self.dino, "get_intermediate_layers"):
            outs = self.dino.get_intermediate_layers(x3, n=4, reshape=False)
            return list(outs)
        ff = self.dino.forward_features(x3)
        toks = ff["x_norm_patchtokens"]
        return [toks, toks, toks, toks]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        assert x.dim() == 4, f"Expected BCHW, got {x.shape}"
        if x.size(1) == 1:
            x3 = x.repeat(1, 3, 1, 1)
        elif x.size(1) == 3:
            x3 = x
        else:
            raise ValueError(f"Input channels must be 1 or 3, got {x.size(1)}")

        h, w = x3.shape[-2:]
        x3 = F.interpolate(x3, size=((h // 14) * 14, (w // 14) * 14), mode="bilinear", align_corners=False)

        tokens_list = self._forward_tokens(x3)
        feats = []
        for i, toks in enumerate(tokens_list):
            if toks.dim() == 3:
                b, n, c = toks.shape
                side = int(n ** 0.5)
                fmap = toks.transpose(1, 2).reshape(b, c, side, side)
            elif toks.dim() == 4:
                fmap = toks
            else:
                raise RuntimeError(f"Unexpected DINO intermediate shape: {tuple(toks.shape)}")
            fmap = self.proj[i](fmap)
            feats.append(fmap)
        return feats
