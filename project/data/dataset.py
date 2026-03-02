from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_img(path: str) -> Tuple[np.ndarray, int]:
    """Load image to float [0,1], HWC; if 1ch then keep HWC with C=1."""
    img = Image.open(path)
    orig_ch = 1 if img.mode in ["L", "I;16", "I"] else 3
    img = img.convert("L") if orig_ch == 1 else img.convert("RGB")
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    arr = arr / 255.0
    return arr, orig_ch


class CleanReconDataset(Dataset):
    """Single-image dataset: return clean image + modality label + path.

    Layout:
      root/IR/clean/*
      root/VI/clean/*
    """

    def __init__(
        self,
        root: str,
        items: Optional[List[Dict]] = None,
        patch_size: Optional[int] = None,
        hflip_p: float = 0.5,
        force_3ch: bool = True,
    ) -> None:
        self.root = Path(root)
        self.patch_size = patch_size
        self.hflip_p = hflip_p
        self.force_3ch = force_3ch
        self.items = items if items is not None else self._scan_items()
        if len(self.items) == 0:
            raise RuntimeError(f"No images found in {root}/IR/clean and {root}/VI/clean")

    def _scan_items(self) -> List[Dict]:
        items: List[Dict] = []
        for mod_name, mod_id in [("IR", 0), ("VI", 1)]:
            d = self.root / mod_name / "clean"
            if not d.exists():
                continue
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    items.append({"path": str(p), "modality_gt": mod_id})
        return sorted(items, key=lambda x: x["path"])

    def __len__(self) -> int:
        return len(self.items)

    def _paired_aug(self, x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        h, w, _ = x.shape
        if self.patch_size is not None and h >= self.patch_size and w >= self.patch_size:
            y0 = rng.randint(0, h - self.patch_size + 1)
            x0 = rng.randint(0, w - self.patch_size + 1)
            x = x[y0:y0 + self.patch_size, x0:x0 + self.patch_size]
        if rng.rand() < self.hflip_p:
            x = x[:, ::-1].copy()
        return x

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        info = self.items[idx]
        clean, orig_ch = _load_img(info["path"])
        rng = np.random.RandomState(idx + 12345)
        clean = self._paired_aug(clean, rng)

        if self.force_3ch and clean.shape[2] == 1:
            clean = np.repeat(clean, 3, axis=2)
        if clean.shape[2] != 3:
            raise RuntimeError(f"Expected 3 channels after conversion, got {clean.shape[2]}")

        x_clean = torch.from_numpy(clean).permute(2, 0, 1).contiguous()
        return {
            "x_clean": x_clean,
            "modality_gt": torch.tensor(info["modality_gt"], dtype=torch.long),
            "path": info["path"],
            "orig_ch": torch.tensor(orig_ch, dtype=torch.long),
        }


def split_items(items: List[Dict], val_ratio: float, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    n_val = max(1, int(len(items) * val_ratio))
    val_idx = set(idx[:n_val].tolist())
    tr, va = [], []
    for i, it in enumerate(items):
        (va if i in val_idx else tr).append(it)
    return tr, va
