from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoEncoder(nn.Module):
    def __init__(self, model_name: str = "dinov2_vitb14", hidden_dim: int = 256):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        embed_dim = self.model.embed_dim
        self.proj = nn.ModuleList([nn.Conv2d(embed_dim, hidden_dim, 1) for _ in range(4)])
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _norm(self, x):
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)

    @torch.no_grad()
    def _extract_tokens(self, x):
        x = self._norm(x)
        return self.model.get_intermediate_layers(x, n=4, reshape=False)

    def extract_multi_layer_feats(self, x: torch.Tensor) -> List[torch.Tensor]:
        assert x.shape[1] == 3, "DINO input must be 3 channels"
        with torch.no_grad():
            toks = self._extract_tokens(x)
        b, _, h, w = x.shape
        ph, pw = h // 14, w // 14
        feats = []
        for i, t in enumerate(toks):
            if isinstance(t, tuple):
                t = t[0]
            if t.shape[1] == ph * pw + 1:
                t = t[:, 1:, :]
            fmap = t.transpose(1, 2).reshape(b, -1, ph, pw).contiguous()
            feats.append(self.proj[i](fmap))
        return feats  # shallow->deep from API ordering
