from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch


def load_image_tensor(path: str, force_3ch: bool = True) -> Tuple[torch.Tensor, int]:
    """Load image to float tensor [0,1], returns (C,H,W) and original channels."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)
    orig_ch = 1 if img.mode in ["L", "I;16", "I"] else 3
    img = img.convert("L") if orig_ch == 1 else img.convert("RGB")
    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[..., None]
    arr = arr.astype(np.float32) / 255.0
    if force_3ch and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor, orig_ch


def save_image_tensor(tensor: torch.Tensor, path: str) -> None:
    """Save (C,H,W) tensor in [0,1] to disk."""
    t = tensor.detach().cpu().float().clamp(0, 1)
    if t.dim() != 3:
        raise ValueError(f"Expected 3D tensor (C,H,W), got shape={tuple(t.shape)}")
    if t.size(0) == 1:
        arr = (t[0].numpy() * 255.0).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    elif t.size(0) == 3:
        arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError(f"Only 1 or 3 channel tensors are supported, got C={t.size(0)}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
