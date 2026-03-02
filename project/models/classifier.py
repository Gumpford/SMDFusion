from typing import Dict
import torch
import torch.nn as nn
import torchvision.models as tvm


class FrozenClassifier(nn.Module):
    """ResNet backbone + three heads wrapper. Input must be (B,3,H,W)."""

    def __init__(self, ckpt_path: str):
        super().__init__()
        net = tvm.resnet18(weights=None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.head_mod = nn.Linear(feat_dim, 2)
        self.head_deg_ir = nn.Linear(feat_dim, 6)
        self.head_deg_vi = nn.Linear(feat_dim, 6)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        missing, unexpected = self.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"[Classifier] missing keys: {len(missing)}")
        if unexpected:
            print(f"[Classifier] unexpected keys: {len(unexpected)}")
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feat = self.backbone(x)
        logits_mod = self.head_mod(feat)
        logits_deg_ir = self.head_deg_ir(feat)
        logits_deg_vi = self.head_deg_vi(feat)
        p_mod = torch.softmax(logits_mod, dim=1).detach()
        p_deg_ir = torch.softmax(logits_deg_ir, dim=1).detach()
        p_deg_vi = torch.softmax(logits_deg_vi, dim=1).detach()
        return {
            "feat": feat.detach(),
            "logits_mod": logits_mod.detach(),
            "logits_deg_ir": logits_deg_ir.detach(),
            "logits_deg_vi": logits_deg_vi.detach(),
            "p_mod": p_mod,
            "p_deg_ir": p_deg_ir,
            "p_deg_vi": p_deg_vi,
            "P": p_mod,
        }
