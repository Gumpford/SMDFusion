from io import BytesIO
import random
import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision.transforms.functional as TF


def _to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return torch.from_numpy(arr).permute(2, 0, 1)


def add_gaussian_noise(x: torch.Tensor, sigma_range=(0.01, 0.08)):
    sigma = random.uniform(*sigma_range)
    return (x + torch.randn_like(x) * sigma).clamp(0, 1)


def gaussian_blur(x: torch.Tensor, r=(0.5, 2.0)):
    k = random.choice([3, 5, 7])
    sigma = random.uniform(*r)
    return TF.gaussian_blur(x, [k, k], [sigma, sigma])


def adjust_contrast(x: torch.Tensor, c=(0.5, 1.5)):
    return TF.adjust_contrast(x, random.uniform(*c)).clamp(0, 1)


def jpeg_compress(x: torch.Tensor, q=(20, 80)):
    img = _to_pil(x)
    try:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*q))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        return _to_tensor(out)
    except Exception:
        levels = random.choice([16, 32, 64])
        return torch.round(x * levels) / levels


def motion_blur_approx(x: torch.Tensor):
    img = _to_pil(x)
    return _to_tensor(img.filter(ImageFilter.BoxBlur(radius=random.uniform(0.8, 2.0))))
