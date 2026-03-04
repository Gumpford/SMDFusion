from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


class EdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[0.05, 0.25, 0.4, 0.25, 0.05]], dtype=torch.float32
        )
        self.register_buffer("k", kernel)
        self.charb = CharbonnierLoss()

    def _conv_gauss(self, img: torch.Tensor) -> torch.Tensor:
        c = img.shape[1]
        k = self.k
        k2d = (k.t() @ k).to(img.dtype).to(img.device)
        k2d = k2d.expand(c, 1, 5, 5)
        return F.conv2d(img, k2d, padding=2, groups=c)

    def _laplacian_kernel(self, current: torch.Tensor) -> torch.Tensor:
        filtered = self._conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        up = torch.zeros_like(filtered)
        up[:, :, ::2, ::2] = down * 4
        filtered = self._conv_gauss(up)
        diff = current - filtered
        return diff

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.charb(self._laplacian_kernel(x), self._laplacian_kernel(y))
