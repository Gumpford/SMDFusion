from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class CleanIRVIDataset(Dataset):
    """Clean-only IR/VI dataset.

    Expected layout:
        root/
          ir/clean/*.png
          vi/clean/*.png
    """

    def __init__(self, root: str, patch_size: int = 256, random_crop: bool = True) -> None:
        self.root = Path(root)
        self.patch_size = patch_size
        self.random_crop = random_crop

        self.samples: List[Tuple[Path, str]] = []
        self.samples.extend((p, "ir") for p in sorted((self.root / "ir" / "clean").glob("*.png")))
        self.samples.extend((p, "vi") for p in sorted((self.root / "vi" / "clean").glob("*.png")))

        if len(self.samples) == 0:
            raise RuntimeError(f"No PNG files found under {self.root}/ir/clean and {self.root}/vi/clean")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")  # IR single-channel is replicated to 3 channels.
        t = TF.to_tensor(img).float()  # [0, 1]
        return t

    def _crop_or_pad(self, t: torch.Tensor) -> torch.Tensor:
        c, h, w = t.shape
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            t = TF.pad(t, [0, 0, pad_w, pad_h], fill=0.0)
            _, h, w = t.shape

        if self.random_crop:
            top = torch.randint(0, h - self.patch_size + 1, (1,)).item()
            left = torch.randint(0, w - self.patch_size + 1, (1,)).item()
        else:
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2

        t = TF.crop(t, top, left, self.patch_size, self.patch_size)
        return t

    def __getitem__(self, idx: int):
        path, modality = self.samples[idx]
        clean = self._read_image(path)
        clean = self._crop_or_pad(clean)
        return {
            "clean": clean,
            "modality": modality,
            "path": str(path),
        }
