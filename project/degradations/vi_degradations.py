from typing import List

import numpy as np

from .common import VI_DEG_NAMES, clip01, jpeg_compress, make_motion_kernel, pil_gaussian_blur

try:
    import cv2
except Exception:
    cv2 = None


def _conv2d_per_channel(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return np.stack([cv2.filter2D(x[..., i], -1, kernel) for i in range(x.shape[2])], axis=2)
    # Fallback: PIL blur approximation if cv2 is unavailable
    return pil_gaussian_blur(x, sigma=1.0)


def apply_vi_degradation(x: np.ndarray, deg_name: str, rng: np.random.RandomState) -> np.ndarray:
    y = x.copy()
    h, w, _ = y.shape

    if deg_name == "lowlight":
        gamma = rng.uniform(1.2, 3.0)
        scale = rng.uniform(0.35, 0.8)
        y = np.power(clip01(y), gamma) * scale
        if rng.rand() < 0.5:
            y += rng.normal(0.0, rng.uniform(0.005, 0.03), size=y.shape).astype(np.float32)
    elif deg_name == "overexposure":
        boost = rng.uniform(1.2, 2.0)
        y = y * boost + rng.uniform(0.02, 0.15)
    elif deg_name == "jpeg":
        quality = int(rng.uniform(8, 45))
        y = jpeg_compress(y, quality)
    elif deg_name == "motion_blur":
        length = int(rng.randint(5, 19))
        angle = float(rng.uniform(0, 180))
        k = make_motion_kernel(length=length, angle_deg=angle)
        y = _conv2d_per_channel(y, k)
    elif deg_name == "haze":
        t = rng.uniform(0.45, 0.85)
        A = rng.uniform(0.7, 1.0)
        y = y * t + A * (1 - t)
    elif deg_name == "rain_snow":
        overlay = np.zeros_like(y)
        n_streaks = rng.randint(100, 450)
        for _ in range(n_streaks):
            xx = rng.randint(0, w)
            yy = rng.randint(0, h)
            length = rng.randint(4, 20)
            for i in range(length):
                xi = np.clip(xx + i // 2, 0, w - 1)
                yi = np.clip(yy + i, 0, h - 1)
                overlay[yi, xi, :] = 1.0
        alpha = rng.uniform(0.05, 0.25)
        y = y * (1 - alpha) + overlay * alpha
    else:
        raise ValueError(f"Unknown VI degradation: {deg_name}")
    return clip01(y)


def sample_vi_degradations(rng: np.random.RandomState) -> List[str]:
    n = 1 if rng.rand() < 0.7 else 2
    idx = rng.choice(len(VI_DEG_NAMES), size=n, replace=False)
    return [VI_DEG_NAMES[i] for i in idx]
