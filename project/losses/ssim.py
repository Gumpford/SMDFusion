import torch
import torch.nn.functional as F


def _gaussian_window(window_size=11, sigma=1.5, channels=3, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)
    return w.repeat(channels, 1, 1, 1)


def ssim(x, y, window_size=11, sigma=1.5):
    c = x.size(1)
    window = _gaussian_window(window_size, sigma, c, x.device, x.dtype)
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=c)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x.pow(2), mu_y.pow(2), mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=c) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8)
    return ssim_map.mean()
