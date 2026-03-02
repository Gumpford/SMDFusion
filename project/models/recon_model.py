from typing import Dict, List

import torch
import torch.nn as nn

from .classifier import FrozenClassifierWrapper
from .dino_encoder import FrozenDINOv2Encoder
from .gated_adapter import GatedAdapter
from .prompt_pool import PromptPool
from .recon_decoder import LightUNetDecoder


class Stage1ReconModel(nn.Module):
    def __init__(
        self,
        classifier_ckpt: str,
        device: torch.device,
        dino_model_name: str = "dinov2_vitb14",
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.classifier = FrozenClassifierWrapper(classifier_ckpt, device=device)
        self.encoder = FrozenDINOv2Encoder(model_name=dino_model_name, hidden_dim=hidden_dim)
        self.prompt_pool = PromptPool(dim=hidden_dim)
        self.adapters = nn.ModuleList([GatedAdapter(dim=hidden_dim, use_dwconv=False) for _ in range(4)])
        self.decoder = LightUNetDecoder(hidden_dim=hidden_dim, out_ch=3)

    def forward(self, x_deg: torch.Tensor, enable_prompt: bool = True) -> Dict[str, torch.Tensor]:
        cls_out = self.classifier(x_deg)
        z, z_aux = self.prompt_pool(cls_out, detach_probs=True)

        feats = self.encoder(x_deg)
        adapted: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        for f, ad in zip(feats, self.adapters):
            f2, g = ad(f, z, enable_prompt=enable_prompt)
            adapted.append(f2)
            gates.append(g)

        x_hat = self.decoder(adapted, out_hw=x_deg.shape[-2:])
        return {
            "x_hat": x_hat,
            "z": z,
            "z_aux": z_aux,
            "gates": gates,
            "probs": {
                "p_mod": cls_out["p_mod"],
                "p_deg_ir": cls_out["p_deg_ir"],
                "p_deg_vi": cls_out["p_deg_vi"],
            },
        }

    def prompt_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "prompt_pool": self.prompt_pool.state_dict(),
            "adapters": self.adapters.state_dict(),
            "encoder_proj": self.encoder.proj.state_dict(),
        }
