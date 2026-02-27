"""Paired reconstruction dataset for degraded/clean grayscale images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset


class PairedReconDataset(Dataset):
    """Load paired grayscale images from root/split/{degraded,clean}."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        crop_size: int | None = None,
        hflip_prob: float = 0.5,
        train: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.train = train
        self.crop_size = crop_size
        self.hflip_prob = hflip_prob

        self.deg_dir = self.root / split / "degraded"
        self.clean_dir = self.root / split / "clean"
        if not self.deg_dir.exists() or not self.clean_dir.exists():
            raise FileNotFoundError(f"Expected {self.deg_dir} and {self.clean_dir} to exist")

        self.samples = []
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        for p in sorted(self.deg_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                clean_p = self.clean_dir / p.name
                if clean_p.exists():
                    self.samples.append((p, clean_p))
        if not self.samples:
            raise ValueError("No paired degraded/clean images found")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _to_tensor_gray(img: Image.Image) -> torch.Tensor:
        arr = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        arr = arr.view(img.size[1], img.size[0]).float() / 255.0
        return arr.unsqueeze(0)

    def _augment(self, x_deg: torch.Tensor, x_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.crop_size is not None:
            h, w = x_deg.shape[-2:]
            cs = min(self.crop_size, h, w)
            if self.train:
                top = torch.randint(0, h - cs + 1, (1,)).item()
                left = torch.randint(0, w - cs + 1, (1,)).item()
            else:
                top = (h - cs) // 2
                left = (w - cs) // 2
            x_deg = x_deg[:, top : top + cs, left : left + cs]
            x_gt = x_gt[:, top : top + cs, left : left + cs]

        if self.train and torch.rand(1).item() < self.hflip_prob:
            x_deg = torch.flip(x_deg, dims=[2])
            x_gt = torch.flip(x_gt, dims=[2])
        return x_deg, x_gt

    def __getitem__(self, idx: int) -> dict[str, Any]:
        deg_path, clean_path = self.samples[idx]
        deg = Image.open(deg_path).convert("L")
        clean = Image.open(clean_path).convert("L")

        x_deg = self._to_tensor_gray(deg)
        x_gt = self._to_tensor_gray(clean)
        x_deg, x_gt = self._augment(x_deg, x_gt)

        return {
            "x_deg": x_deg,
            "x_gt": x_gt,
            "meta": {"name": deg_path.name, "deg_path": str(deg_path), "clean_path": str(clean_path)},
        }
