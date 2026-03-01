from typing import Dict, List

import torch
import torch.nn as nn

from .dino_encoder import FrozenDINOv2Encoder
from .gated_adapter import GatedAdapter
from .prompt_pool import PromptPool
from .recon_decoder import LightUNetDecoder


class Stage1ReconModel(nn.Module):
    def __init__(self, dino_model_name: str = "dinov2_vitb14", out_ch: int = 3, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = FrozenDINOv2Encoder(model_name=dino_model_name, hidden_dim=hidden_dim)
        self.prompt_pool = PromptPool(dim=hidden_dim)
        self.adapters = nn.ModuleList([GatedAdapter(dim=hidden_dim) for _ in range(4)])
        self.decoder = LightUNetDecoder(hidden_dim=hidden_dim, out_ch=out_ch)

    def forward(
        self,
        x_deg: torch.Tensor,
        cls_outputs: Dict[str, torch.Tensor],
        enable_prompt: bool = True,
    ) -> Dict[str, torch.Tensor]:
        feats = self.encoder(x_deg)  # list of 4 maps
        z, z_aux = self.prompt_pool(cls_outputs=cls_outputs, detach_probs=True)

        adapted: List[torch.Tensor] = []
        gates = []
        for f, ad in zip(feats, self.adapters):
            f2, g = ad(f, z, enable_prompt=enable_prompt)
            adapted.append(f2)
            gates.append(g)

        x_hat = self.decoder(adapted, out_hw=x_deg.shape[-2:])
        return {"x_hat": x_hat, "z": z, "z_aux": z_aux, "gates": gates}

    def prompt_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "prompt_pool": self.prompt_pool.state_dict(),
            "adapters": self.adapters.state_dict(),
            "encoder_proj": self.encoder.proj.state_dict(),
        }


def freeze_classifier(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_frozen_classifier(ckpt_path: str, device: torch.device) -> nn.Module:
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, nn.Module):
        return freeze_classifier(obj.to(device))
    if isinstance(obj, dict):
        for key in ["model", "classifier", "net"]:
            if key in obj and isinstance(obj[key], nn.Module):
                return freeze_classifier(obj[key].to(device))
    raise RuntimeError(
        "Unable to load classifier nn.Module from checkpoint. "
        "Please provide a checkpoint saved as nn.Module or dict with key model/classifier/net."
    )
