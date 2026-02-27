"""Frozen DINOv2 encoder + trainable decoder reconstruction network."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gated_modulation import GatedModulation


class FrozenDINOv2Encoder(nn.Module):
    """DINOv2 feature extractor returning three scales (high/mid/low)."""

    def __init__(self, model_name: str = "dinov2_vits14", local_ckpt: str = "") -> None:
        super().__init__()
        self.model_name = model_name
        self.model = self._build_model(model_name, local_ckpt)
        self.patch_size = 14
        self.embed_dim = self._infer_embed_dim()
        self._freeze()

    def _build_model(self, model_name: str, local_ckpt: str) -> nn.Module:
        if local_ckpt:
            payload = torch.load(local_ckpt, map_location="cpu")
            if isinstance(payload, nn.Module):
                return payload
            raise ValueError("local_ckpt must contain serialized nn.Module for encoder")
        try:
            return torch.hub.load("facebookresearch/dinov2", model_name)
        except Exception as exc:  # pragma: no cover - runtime fallback
            warnings.warn(f"Failed to load DINOv2 from torch.hub ({exc}). Falling back to tiny encoder.")
            return nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.GELU(),
            )

    def _infer_embed_dim(self) -> int:
        for attr in ["embed_dim", "num_features"]:
            if hasattr(self.model, attr):
                return int(getattr(self.model, attr))
        return 256

    def _freeze(self) -> None:
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _tokens_to_map(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = tokens.shape
        hh = h // self.patch_size
        ww = w // self.patch_size
        if hh * ww != n:
            side = int(n**0.5)
            hh = side
            ww = side
        fmap = tokens.transpose(1, 2).reshape(b, c, hh, ww)
        return fmap

    @torch.no_grad()
    def forward(self, x_rgb: torch.Tensor) -> list[torch.Tensor]:
        b, _, h, w = x_rgb.shape

        if hasattr(self.model, "get_intermediate_layers"):
            feats = self.model.get_intermediate_layers(x_rgb, n=3, reshape=False)
            fm_list = [self._tokens_to_map(t, h, w) for t in feats]
        else:
            feat = self.model(x_rgb)
            if feat.ndim == 4:
                fm_list = [feat]
            elif feat.ndim == 3:
                fm_list = [self._tokens_to_map(feat, h, w)]
            else:
                raise ValueError("Unsupported encoder output shape")

        f_high = fm_list[0]
        f_mid = F.avg_pool2d(f_high, kernel_size=2, stride=2)
        f_low = F.avg_pool2d(f_mid, kernel_size=2, stride=2)
        return [f_high, f_mid, f_low]


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TrainableDecoder(nn.Module):
    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.low = _DecoderBlock(c3, c2)
        self.mid = _DecoderBlock(c2 + c2, c1)
        self.high = _DecoderBlock(c1 + c1, c1)
        self.out = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, feats: list[torch.Tensor], out_hw: tuple[int, int]) -> torch.Tensor:
        f1, f2, f3 = feats
        x = self.low(f3)
        x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.mid(torch.cat([x, f2], dim=1))
        x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.high(torch.cat([x, f1], dim=1))
        x = self.out(x)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


class DINOv2AutoEncoder(nn.Module):
    """Prompt-modulated reconstruction network with frozen encoder."""

    def __init__(self, prompt_dim: int, dinov2_name: str = "dinov2_vits14", dinov2_local_ckpt: str = "") -> None:
        super().__init__()
        self.encoder = FrozenDINOv2Encoder(dinov2_name, local_ckpt=dinov2_local_ckpt)
        c = self.encoder.embed_dim
        self.channels = [c, c, c]
        self.modulator = GatedModulation(prompt_dim=prompt_dim, channels=self.channels)
        self.decoder = TrainableDecoder(self.channels)

    def freeze_encoder(self) -> None:
        self.encoder._freeze()

    def trainable_parameters(self):
        for p in self.modulator.parameters():
            yield p
        for p in self.decoder.parameters():
            yield p

    def forward(self, x_degraded: torch.Tensor, prompt_vec: torch.Tensor) -> torch.Tensor:
        if x_degraded.ndim != 4 or x_degraded.size(1) != 1:
            raise ValueError("x_degraded must have shape (B,1,H,W)")
        if prompt_vec.ndim != 2:
            raise ValueError("prompt_vec must have shape (B,D)")

        x_rgb = x_degraded.repeat(1, 3, 1, 1)
        feats = self.encoder(x_rgb)
        feats_mod = self.modulator(feats, prompt_vec)
        y_hat = self.decoder(feats_mod, out_hw=x_degraded.shape[-2:])
        return y_hat
