from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from degradations.ir_degradations import apply_ir_degradation, sample_ir_degradations
from degradations.vi_degradations import apply_vi_degradation, sample_vi_degradations


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_img(path: str, force_3ch: bool = True) -> Tuple[np.ndarray, int]:
    img = Image.open(path)
    orig_ch = 1 if img.mode in ["L", "I;16", "I"] else 3
    img = img.convert("L") if orig_ch == 1 else img.convert("RGB")
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    arr /= 255.0
    if force_3ch and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr, orig_ch


class CleanToDegradedReconDataset(Dataset):
    """Loads clean IR/VI images and applies online modality-specific degradations."""

    def __init__(
        self,
        root: str,
        items: Optional[List[Dict]] = None,
        crop_size: Optional[int] = None,
        hflip_p: float = 0.5,
        seed: int = 42,
        force_3ch: bool = True,
    ) -> None:
        self.root = Path(root)
        self.crop_size = crop_size
        self.hflip_p = hflip_p
        self.base_seed = int(seed)
        self.force_3ch = force_3ch

        self.items = items if items is not None else self._scan_items()
        if len(self.items) == 0:
            raise RuntimeError(f"No images found under {root}/IR/clean or {root}/VI/clean")

    def _scan_items(self) -> List[Dict]:
        """
        Supported layouts:
        1) Split by modality (recommended):
           root/IR/clean/*.png
           root/VI/clean/*.png
        2) Mixed clean folder:
           root/clean/**/*.{png,jpg,...}
           modality inferred from parent folder or filename token (ir/vi).
        """
        items: List[Dict] = []

        # Layout-1: explicit modality folders
        for mod in ["IR", "VI"]:
            clean_dir = self.root / mod / "clean"
            if not clean_dir.exists():
                continue
            for p in clean_dir.iterdir():
                if p.suffix.lower() in IMG_EXTS:
                    items.append({"path": str(p), "modality": mod})

        if items:
            return sorted(items, key=lambda x: x["path"])

        # Layout-2: mixed clean folder, infer modality
        mixed_clean = self.root / "clean"
        if mixed_clean.exists():
            for p in mixed_clean.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
                    continue
                mod = self._infer_modality_from_path(p)
                items.append({"path": str(p), "modality": mod})

        return sorted(items, key=lambda x: x["path"])

    @staticmethod
    def _infer_modality_from_path(path: Path) -> str:
        """Infer modality from parent folder names or filename tokens.

        Accepted cues (case-insensitive):
        - parent folder contains 'ir' or 'vi'
        - filename stem contains tokens separated by [_-.] with 'ir' or 'vi'
        """
        parts = [p.lower() for p in path.parts]
        if any(part == "ir" for part in parts):
            return "IR"
        if any(part == "vi" for part in parts):
            return "VI"

        stem = path.stem.lower().replace("-", "_").replace(".", "_")
        tokens = stem.split("_")
        if "ir" in tokens:
            return "IR"
        if "vi" in tokens:
            return "VI"

        raise RuntimeError(
            "Cannot infer modality for mixed-layout sample: "
            f"{path}. Please either use root/IR/clean & root/VI/clean, "
            "or include modality cues in parent folder/file name (ir/vi)."
        )

    def __len__(self) -> int:
        return len(self.items)

    def _paired_aug(self, clean: np.ndarray, deg: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        assert clean.shape == deg.shape
        h, w, _ = clean.shape
        if self.crop_size is not None and h >= self.crop_size and w >= self.crop_size:
            y0 = rng.randint(0, h - self.crop_size + 1)
            x0 = rng.randint(0, w - self.crop_size + 1)
            clean = clean[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
            deg = deg[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
        if rng.rand() < self.hflip_p:
            clean = clean[:, ::-1].copy()
            deg = deg[:, ::-1].copy()
        return clean, deg

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        info = self.items[idx]
        clean, orig_ch = _load_img(info["path"], force_3ch=self.force_3ch)

        rng = np.random.RandomState(self.base_seed + idx)
        if info["modality"] == "IR":
            names = sample_ir_degradations(rng)
            deg = clean.copy()
            for n in names:
                deg = apply_ir_degradation(deg, n, rng)
        else:
            names = sample_vi_degradations(rng)
            deg = clean.copy()
            for n in names:
                deg = apply_vi_degradation(deg, n, rng)

        clean, deg = self._paired_aug(clean, deg, rng)
        clean_t = torch.from_numpy(clean).permute(2, 0, 1).contiguous()
        deg_t = torch.from_numpy(deg).permute(2, 0, 1).contiguous()

        return {
            "x_clean": clean_t,
            "x_deg": deg_t,
            "modality": info["modality"],
            "path": info["path"],
            "orig_ch": torch.tensor(orig_ch, dtype=torch.long),
        }


def split_items(items: List[Dict], val_ratio: float, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    n_val = max(1, int(len(items) * val_ratio))
    val_idx = set(idx[:n_val].tolist())
    train_items, val_items = [], []
    for i, it in enumerate(items):
        (val_items if i in val_idx else train_items).append(it)
    return train_items, val_items
