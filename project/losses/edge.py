import torch
import torch.nn.functional as F


def sobel_edge_loss(x, y):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    kx = kx.repeat(x.shape[1], 1, 1, 1)
    ky = ky.repeat(x.shape[1], 1, 1, 1)
    gx1 = F.conv2d(x, kx, padding=1, groups=x.shape[1])
    gy1 = F.conv2d(x, ky, padding=1, groups=x.shape[1])
    gx2 = F.conv2d(y, kx, padding=1, groups=y.shape[1])
    gy2 = F.conv2d(y, ky, padding=1, groups=y.shape[1])
    return (gx1 - gx2).abs().mean() + (gy1 - gy2).abs().mean()
