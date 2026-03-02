import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, channels: int = 3, device="cpu") -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return w2d.repeat(channels, 1, 1, 1)


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5, c1: float = 0.01**2, c2: float = 0.03**2):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.c1 = c1
        self.c2 = c2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape
        b, c, _, _ = x.shape
        win = _gaussian_window(self.window_size, self.sigma, c, device=x.device).type_as(x)

        mu_x = F.conv2d(x, win, padding=self.window_size // 2, groups=c)
        mu_y = F.conv2d(y, win, padding=self.window_size // 2, groups=c)

        sigma_x = F.conv2d(x * x, win, padding=self.window_size // 2, groups=c) - mu_x ** 2
        sigma_y = F.conv2d(y * y, win, padding=self.window_size // 2, groups=c) - mu_y ** 2
        sigma_xy = F.conv2d(x * y, win, padding=self.window_size // 2, groups=c) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)) / (
            (mu_x ** 2 + mu_y ** 2 + self.c1) * (sigma_x + sigma_y + self.c2) + 1e-8
        )
        ssim_val = ssim_map.mean()
        return 1.0 - ssim_val
