import random
import torch
from .common import add_gaussian_noise, gaussian_blur, adjust_contrast

IR_DEGS = ["noise", "blur", "contrast", "stripe_fpn", "dead_pixels", "gain_offset"]


def stripe_fpn(x: torch.Tensor):
    c, h, w = x.shape
    out = x.clone()
    step = random.randint(8, 32)
    for i in range(0, w, step):
        out[:, :, i:i+1] = (out[:, :, i:i+1] + random.uniform(-0.2, 0.2)).clamp(0, 1)
    return out


def dead_pixels(x: torch.Tensor):
    out = x.clone()
    p = random.uniform(0.001, 0.01)
    mask = torch.rand_like(out[0]) < p
    out[:, mask] = 0.0
    return out


def gain_offset(x: torch.Tensor):
    gain = random.uniform(0.8, 1.2)
    bias = random.uniform(-0.1, 0.1)
    return (x * gain + bias).clamp(0, 1)


def apply_ir_degradation(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "noise":
        return add_gaussian_noise(x)
    if name == "blur":
        return gaussian_blur(x)
    if name == "contrast":
        return adjust_contrast(x)
    if name == "stripe_fpn":
        return stripe_fpn(x)
    if name == "dead_pixels":
        return dead_pixels(x)
    if name == "gain_offset":
        return gain_offset(x)
    raise ValueError(f"Unknown IR degradation: {name}")


def sample_ir_degradation(x: torch.Tensor) -> torch.Tensor:
    n = random.randint(1, 2)
    ops = random.sample(IR_DEGS, n)
    y = x
    for op in ops:
        y = apply_ir_degradation(y, op)
    return y
