from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import torch


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path)
    if img.mode not in ["RGB", "L"]:
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def save_tensor_image(x: torch.Tensor, path: str, out_ch: int = 3) -> None:
    assert out_ch in [1, 3], f"out_ch must be 1 or 3, got {out_ch}"
    x = x.detach().cpu().clamp(0, 1)
    if x.dim() == 4:
        x = x[0]
    if out_ch == 1:
        x = x[0:1]
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    if out_ch == 1:
        arr = arr[..., 0]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def list_images(path: str):
    p = Path(path)
    if p.is_file():
        return [p]
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    out = []
    for e in exts:
        out.extend(sorted(p.rglob(e)))
    return out
