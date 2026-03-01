import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        assert pred.shape == gt.shape
        c = pred.size(1)
        kx = self.kx.repeat(c, 1, 1, 1)
        ky = self.ky.repeat(c, 1, 1, 1)

        pred_gx = F.conv2d(pred, kx, padding=1, groups=c)
        pred_gy = F.conv2d(pred, ky, padding=1, groups=c)
        gt_gx = F.conv2d(gt, kx, padding=1, groups=c)
        gt_gy = F.conv2d(gt, ky, padding=1, groups=c)
        return F.l1_loss(pred_gx, gt_gx) + F.l1_loss(pred_gy, gt_gy)
