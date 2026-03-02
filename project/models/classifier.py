from typing import Dict

import torch
import torch.nn as nn


class FrozenClassifierWrapper(nn.Module):
    """Load and run a frozen classifier that outputs required probability heads.

    Expected forward output dict keys:
      - p_mod: (B,2)
      - p_deg_ir: (B,6)
      - p_deg_vi: (B,6)
    """

    def __init__(self, ckpt_path: str, device: torch.device) -> None:
        super().__init__()
        obj = torch.load(ckpt_path, map_location=device)
        model = None
        if isinstance(obj, nn.Module):
            model = obj
        elif isinstance(obj, dict):
            for k in ["model", "classifier", "net"]:
                if k in obj and isinstance(obj[k], nn.Module):
                    model = obj[k]
                    break
        if model is None:
            raise RuntimeError(
                "Cannot parse classifier model from checkpoint. "
                "Supported formats: nn.Module or dict with key model/classifier/net."
            )

        self.model = model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"Classifier input must be BCHW, got {tuple(x.shape)}")
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.size(1) != 3:
            raise ValueError(f"Classifier expects 1 or 3 channels, got {x.size(1)}")

        out = self.model(x)
        required = ["p_mod", "p_deg_ir", "p_deg_vi"]
        for k in required:
            if k not in out:
                raise KeyError(f"Classifier output missing key: {k}")
        out["p_mod"] = out["p_mod"].detach()
        out["p_deg_ir"] = out["p_deg_ir"].detach()
        out["p_deg_vi"] = out["p_deg_vi"].detach()
        return out
