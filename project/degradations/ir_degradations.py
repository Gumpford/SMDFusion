from typing import List, Tuple

import numpy as np

from .common import IR_DEG_NAMES, clip01, pil_gaussian_blur


def _stripe_pattern(h: int, w: int, rng: np.random.RandomState) -> np.ndarray:
    vertical = rng.rand() > 0.5
    freq = rng.randint(4, 32)
    amp = rng.uniform(0.02, 0.15)
    axis = np.arange(w if vertical else h)
    pattern = amp * np.sin(2 * np.pi * axis / max(freq, 1))
    if vertical:
        pat = np.tile(pattern[None, :], (h, 1))
    else:
        pat = np.tile(pattern[:, None], (1, w))
    return pat


def apply_ir_degradation(x: np.ndarray, deg_name: str, rng: np.random.RandomState) -> np.ndarray:
    """x: HWC float32 in [0,1]."""
    y = x.copy()
    h, w, c = y.shape

    if deg_name == "noise":
        sigma = rng.uniform(0.01, 0.12)
        y += rng.normal(0.0, sigma, size=y.shape).astype(np.float32)
    elif deg_name == "blur":
        sigma = rng.uniform(0.5, 2.2)
        y = pil_gaussian_blur(y, sigma=sigma)
    elif deg_name == "contrast":
        gamma = rng.uniform(0.6, 1.8)
        scale = rng.uniform(0.7, 1.3)
        y = np.power(clip01(y), gamma) * scale
    elif deg_name == "stripe_fpn":
        pat = _stripe_pattern(h, w, rng)
        y += pat[..., None]
    elif deg_name == "dead_pixels":
        rate = rng.uniform(0.001, 0.02)
        mask = rng.rand(h, w, 1) < rate
        val = rng.choice([0.0, 1.0])
        y[mask.repeat(c, axis=2)] = val
    elif deg_name == "gain_offset":
        gain = rng.uniform(0.7, 1.4)
        offset = rng.uniform(-0.12, 0.12)
        y = gain * y + offset
    else:
        raise ValueError(f"Unknown IR degradation: {deg_name}")
    return clip01(y)


def sample_ir_degradations(rng: np.random.RandomState) -> List[str]:
    n = 1 if rng.rand() < 0.7 else 2
    idx = rng.choice(len(IR_DEG_NAMES), size=n, replace=False)
    return [IR_DEG_NAMES[i] for i in idx]
