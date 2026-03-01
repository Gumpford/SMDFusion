from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


IR_DEG_NAMES = ["noise", "blur", "contrast", "stripe_fpn", "dead_pixels", "gain_offset"]
VI_DEG_NAMES = ["lowlight", "overexposure", "jpeg", "motion_blur", "haze", "rain_snow"]


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def pil_gaussian_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    img = Image.fromarray((clip01(x) * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    y = np.array(img).astype(np.float32) / 255.0
    if y.ndim == 2:
        y = y[..., None]
    return y


def jpeg_compress(x: np.ndarray, quality: int = 30) -> np.ndarray:
    quality = int(np.clip(quality, 5, 95))
    if x.shape[2] == 1:
        img = Image.fromarray((clip01(x[..., 0]) * 255).astype(np.uint8), mode="L")
    else:
        img = Image.fromarray((clip01(x) * 255).astype(np.uint8), mode="RGB")
    try:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        y = np.array(Image.open(buf)).astype(np.float32) / 255.0
    except Exception:
        bins = max(8, quality // 5)
        y = np.round(clip01(x) * bins) / bins
    if y.ndim == 2:
        y = y[..., None]
    return y


def make_motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), dtype=np.float32)
    k[length // 2, :] = 1.0
    img = Image.fromarray((k * 255).astype(np.uint8))
    img = img.rotate(angle_deg)
    k = np.array(img).astype(np.float32) / 255.0
    s = k.sum()
    return k / (s + 1e-8)
