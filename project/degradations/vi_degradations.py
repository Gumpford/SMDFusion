import random
import torch
from .common import jpeg_compress, motion_blur_approx, adjust_contrast

VI_DEGS = ["lowlight", "overexposure", "jpeg", "motion_blur", "haze", "rain_snow"]


def lowlight(x):
    return (x * random.uniform(0.2, 0.7)).clamp(0, 1)


def overexposure(x):
    return (x * random.uniform(1.2, 1.8)).clamp(0, 1)


def haze(x):
    a = random.uniform(0.6, 0.9)
    return (x * a + (1 - a)).clamp(0, 1)


def rain_snow(x):
    out = x.clone()
    _, h, w = out.shape
    num = random.randint(50, 200)
    for _ in range(num):
        i, j = random.randint(0, h-1), random.randint(0, w-1)
        out[:, max(0, i-1):min(h, i+2), max(0, j-1):min(w, j+2)] = 1.0
    return out.clamp(0, 1)


def apply_vi_degradation(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "lowlight":
        return lowlight(x)
    if name == "overexposure":
        return overexposure(x)
    if name == "jpeg":
        return jpeg_compress(x)
    if name == "motion_blur":
        return motion_blur_approx(x)
    if name == "haze":
        return haze(x)
    if name == "rain_snow":
        return rain_snow(x)
    raise ValueError(f"Unknown VI degradation: {name}")


def sample_vi_degradation(x: torch.Tensor) -> torch.Tensor:
    n = random.randint(1, 2)
    ops = random.sample(VI_DEGS, n)
    y = x
    for op in ops:
        y = apply_vi_degradation(y, op)
    return y
