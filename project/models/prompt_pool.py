from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptPool(nn.Module):
    """Hierarchical prompt pool with modality/degradation prompts."""

    def __init__(self, dim: int = 256, init_std: float = 0.02, temperature: float = 1.0) -> None:
        super().__init__()
        self.M = nn.Parameter(torch.randn(2, dim) * init_std)
        self.D_ir = nn.Parameter(torch.randn(6, dim) * init_std)
        self.D_vi = nn.Parameter(torch.randn(6, dim) * init_std)
        self.prompt_scale = nn.Parameter(torch.tensor(1e-3))
        self.temperature = temperature

        self.composer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def orth_loss(self) -> torch.Tensor:
        loss = 0.0
        for P in [self.D_ir, self.D_vi]:
            p = F.normalize(P, dim=1)
            gram = p.T @ p
            I = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
            loss = loss + ((gram - I) ** 2).mean()
        return loss

    def forward(self, cls_outputs: Dict[str, torch.Tensor], detach_probs: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        p_mod = cls_outputs["p_mod"]
        p_deg_ir = cls_outputs["p_deg_ir"]
        p_deg_vi = cls_outputs["p_deg_vi"]
        if detach_probs:
            p_mod = p_mod.detach()
            p_deg_ir = p_deg_ir.detach()
            p_deg_vi = p_deg_vi.detach()

        T = self.temperature
        p_mod = F.softmax(torch.log(p_mod + 1e-8) / T, dim=-1)
        p_deg_ir = F.softmax(torch.log(p_deg_ir + 1e-8) / T, dim=-1)
        p_deg_vi = F.softmax(torch.log(p_deg_vi + 1e-8) / T, dim=-1)

        z_m = p_mod @ self.M
        z_d_ir = p_deg_ir @ self.D_ir
        z_d_vi = p_deg_vi @ self.D_vi

        w_ir = p_mod[:, 0:1]
        w_vi = p_mod[:, 1:2]
        z_d = w_ir * z_d_ir + w_vi * z_d_vi

        z = self.composer(torch.cat([z_m, z_d], dim=-1))
        z = self.prompt_scale * z

        aux = {"z_m": z_m, "z_d": z_d, "z_d_ir": z_d_ir, "z_d_vi": z_d_vi}
        return z, aux
