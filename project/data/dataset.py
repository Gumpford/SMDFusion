from pathlib import Path
import random
from typing import Tuple
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class MixedCleanDataset(Dataset):
    def __init__(self, root: str, patch_size: int = 0, hflip: bool = True):
        self.root = Path(root)
        self.patch_size = patch_size
        self.hflip = hflip
        self.samples = []
        for mod, lab in [("IR", 0), ("VI", 1)]:
            d = self.root / mod / "clean"
            if not d.exists():
                continue
            for p in sorted(d.glob("*")):
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                    self.samples.append((str(p), lab))
        if not self.samples:
            raise RuntimeError(f"No images found under {root}/IR(clean) or VI(clean)")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load(path: str) -> torch.Tensor:
        img = Image.open(path)
        if img.mode not in ["RGB", "L"]:
            img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx: int):
        path, modality = self.samples[idx]
        x = self._load(path)
        c, h, w = x.shape
        if self.patch_size > 0 and h >= self.patch_size and w >= self.patch_size:
            i = random.randint(0, h - self.patch_size)
            j = random.randint(0, w - self.patch_size)
            x = x[:, i:i + self.patch_size, j:j + self.patch_size]
        if self.hflip and random.random() < 0.5:
            x = TF.hflip(x)
        return x, torch.tensor(modality, dtype=torch.long), path
