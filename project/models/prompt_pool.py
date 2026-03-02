import torch
import torch.nn as nn


class PromptPool(nn.Module):
    def __init__(self, d: int = 256):
        super().__init__()
        self.M = nn.Parameter(torch.randn(2, d) * 0.02)
        self.D_ir = nn.Parameter(torch.randn(6, d) * 0.02)
        self.D_vi = nn.Parameter(torch.randn(6, d) * 0.02)
        self.prompt_scale = nn.Parameter(torch.tensor(1e-3))
        self.composer = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.norm = nn.LayerNorm(d)

    def compose(self, p_mod, p_deg_ir, p_deg_vi, temperature: float = 1.0):
        p_mod = torch.softmax(torch.log(p_mod.clamp_min(1e-8)) / temperature, dim=-1)
        p_deg_ir = torch.softmax(torch.log(p_deg_ir.clamp_min(1e-8)) / temperature, dim=-1)
        p_deg_vi = torch.softmax(torch.log(p_deg_vi.clamp_min(1e-8)) / temperature, dim=-1)
        z_m = p_mod @ self.M
        z_ir = p_deg_ir @ self.D_ir
        z_vi = p_deg_vi @ self.D_vi
        z_d = p_mod[:, 0:1] * z_ir + p_mod[:, 1:2] * z_vi
        z = self.norm(self.composer(torch.cat([z_m, z_d], dim=1))) * self.prompt_scale
        return z

    def orth_loss(self):
        def _orth(P):
            Pn = P / (P.norm(dim=1, keepdim=True) + 1e-6)
            I = torch.eye(Pn.shape[1], device=Pn.device, dtype=Pn.dtype)
            return ((Pn.t() @ Pn - I) ** 2).sum()

        return _orth(self.D_ir) + _orth(self.D_vi)
