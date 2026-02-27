"""Hierarchical modality/degradation predictor for prompt gating."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyImageEncoder(nn.Module):
    """Fallback lightweight image encoder when CLIP/VFM is not used."""

    def __init__(self, out_dim: int = 512) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(images).flatten(1)
        return self.fc(feat)


@dataclass
class PredictorConfig:
    feat_dim: int
    kir: int
    kvi: int
    hidden_dim: int = 512
    temperature: float = 1.0
    label_smoothing: float = 0.0
    use_clip: bool = False
    clip_model_name: str = "ViT-H-14"
    freeze_clip: bool = True


class ClipVisualEncoder(nn.Module):
    """Wrapper for open_clip visual encoder outputting global image features."""

    def __init__(self, model_name: str, freeze_clip: bool = True) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "use_clip=True requires open_clip_torch. Install it or set --use_clip false."
            ) from exc

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained="laion2b_s32b_b79k",
        )
        self.model = model.visual
        if freeze_clip:
            self.freeze_clip()

    def freeze_clip(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


class HierarchicalPromptPredictor(nn.Module):
    """Predict modality and per-modality degradation, then synthesize global gating P."""

    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.kir = config.kir
        self.kvi = config.kvi
        self.feat_dim = config.feat_dim
        self.temperature = config.temperature
        self.label_smoothing = config.label_smoothing

        self.encoder = self._build_encoder()
        self.fc1 = nn.Linear(self.feat_dim, config.hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.modality_head = nn.Linear(config.hidden_dim, 2)
        self.deg_head_ir = nn.Linear(config.hidden_dim, self.kir)
        self.deg_head_vi = nn.Linear(config.hidden_dim, self.kvi)

    def _build_encoder(self) -> nn.Module:
        if self.config.use_clip:
            return ClipVisualEncoder(self.config.clip_model_name, freeze_clip=self.config.freeze_clip)
        return TinyImageEncoder(out_dim=self.feat_dim)

    def freeze_clip(self) -> None:
        """Compatibility helper similar to common classifier code style."""
        if isinstance(self.encoder, ClipVisualEncoder):
            self.encoder.freeze_clip()

    def _extract_feat(
        self,
        images: torch.Tensor | None = None,
        feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (images is None) == (feat is None):
            raise ValueError("Provide exactly one of images or feat")

        if feat is None:
            if images is None or images.ndim != 4:
                raise ValueError("images must be Tensor(B,3,H,W)")
            feat = self.encoder(images)

        if feat.ndim != 2:
            raise ValueError(f"Expected feature shape (B,C), got {tuple(feat.shape)}")
        assert feat.size(1) == self.feat_dim, f"feat dim mismatch: {feat.size(1)} vs {self.feat_dim}"
        return feat

    def _predict_logits(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.fc2(self.drop(self.relu(self.fc1(feat))))
        logits_mod = self.modality_head(hidden)
        logits_deg_ir = self.deg_head_ir(hidden)
        logits_deg_vi = self.deg_head_vi(hidden)
        return logits_mod, logits_deg_ir, logits_deg_vi

    def _stable_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        scaled = logits / max(self.temperature, 1e-6)
        scaled = scaled - scaled.max(dim=-1, keepdim=True).values
        probs = F.softmax(scaled, dim=-1)
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            num_classes = probs.size(-1)
            probs = (1.0 - eps) * probs + eps / num_classes
        return probs

    def _compose_gating(
        self,
        p_mod: torch.Tensor,
        p_deg_ir: torch.Tensor,
        p_deg_vi: torch.Tensor,
    ) -> torch.Tensor:
        p_ir = p_mod[:, 0:1] * p_deg_ir
        p_vi = p_mod[:, 1:2] * p_deg_vi
        p_global = torch.cat([p_ir, p_vi], dim=-1)
        return p_global

    def forward(
        self,
        images: torch.Tensor | None = None,
        feat: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        feat = self._extract_feat(images=images, feat=feat)
        logits_mod, logits_deg_ir, logits_deg_vi = self._predict_logits(feat)

        p_mod = self._stable_softmax(logits_mod)
        p_deg_ir = self._stable_softmax(logits_deg_ir)
        p_deg_vi = self._stable_softmax(logits_deg_vi)
        p_global = self._compose_gating(p_mod, p_deg_ir, p_deg_vi)

        assert logits_mod.shape[1] == 2
        assert logits_deg_ir.shape[1] == self.kir
        assert logits_deg_vi.shape[1] == self.kvi
        assert p_global.shape == (feat.size(0), self.kir + self.kvi)

        outputs = {
            "feat": feat,
            "logits_mod": logits_mod,
            "logits_deg_ir": logits_deg_ir,
            "logits_deg_vi": logits_deg_vi,
            "p_mod": p_mod,
            "p_deg_ir": p_deg_ir,
            "p_deg_vi": p_deg_vi,
            "P": p_global,
        }
        if return_logits:
            return outputs
        return outputs

    @torch.no_grad()
    def predict_P(self, images: torch.Tensor) -> torch.Tensor:
        """Convenience inference API to return only global gating P."""
        was_training = self.training
        self.eval()
        out = self.forward(images=images)
        if was_training:
            self.train()
        return out["P"]
