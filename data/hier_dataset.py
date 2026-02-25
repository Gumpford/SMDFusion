"""Dataset for hierarchical modality/degradation prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class HierarchicalDegradationDataset(Dataset):
    """Load images from root/split/{ir,vi}/<deg_name>/*.png"""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Any = None,
        exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.exts = tuple(e.lower() for e in exts)
        self.samples: list[tuple[Path, int, int, str]] = []

        split_dir = self.root / split
        ir_root = split_dir / "ir"
        vi_root = split_dir / "vi"
        if not ir_root.exists() or not vi_root.exists():
            raise FileNotFoundError(f"Expected {ir_root} and {vi_root} to exist")

        self.ir_degs = sorted([p.name for p in ir_root.iterdir() if p.is_dir()])
        self.vi_degs = sorted([p.name for p in vi_root.iterdir() if p.is_dir()])
        self.Kir = len(self.ir_degs)
        self.Kvi = len(self.vi_degs)
        if self.Kir == 0 or self.Kvi == 0:
            raise ValueError("Both IR and VI degradations must have at least one class")

        self.ir_map = {name: idx for idx, name in enumerate(self.ir_degs)}
        self.vi_map = {name: idx for idx, name in enumerate(self.vi_degs)}
        self._scan_modality(ir_root, modality=0, label_map=self.ir_map)
        self._scan_modality(vi_root, modality=1, label_map=self.vi_map)

        if len(self.samples) == 0:
            raise ValueError("No images found in dataset")

    def _scan_modality(self, root: Path, modality: int, label_map: dict[str, int]) -> None:
        modality_name = "ir" if modality == 0 else "vi"
        for deg_name, local_label in label_map.items():
            class_dir = root / deg_name
            for p in class_dir.iterdir():
                if p.is_file() and p.suffix.lower() in self.exts:
                    self.samples.append((p, modality, local_label, modality_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path, y_mod, y_deg, modality_name = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(img)
        else:
            image = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        out = {
            "image": image,
            "y_mod": torch.tensor(y_mod, dtype=torch.long),
            "y_deg": torch.tensor(y_deg, dtype=torch.long),
            "meta": {
                "path": str(path),
                "modality_name": modality_name,
                "deg_name": self.ir_degs[y_deg] if y_mod == 0 else self.vi_degs[y_deg],
            },
        }
        return out

    def get_label_maps(self) -> dict[str, Any]:
        return {
            "ir": {name: i for i, name in enumerate(self.ir_degs)},
            "vi": {name: i for i, name in enumerate(self.vi_degs)},
            "Kir": self.Kir,
            "Kvi": self.Kvi,
        }
