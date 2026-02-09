"""ImageFolderDataset for self-supervised degradation modeling."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image, ImageFilter, ImageEnhance

from datasets.base_dataset import BaseDataset


class ImageFolderDataset(BaseDataset):
    """Load single images from modality folders and apply random degradations."""

    def __init__(self, root: str, split: str) -> None:
        super().__init__(root, split)
        self.samples: List[Tuple[str, int]] = []
        split_root = os.path.join(root, split)
        for modality, modality_id in ("IR", 0), ("VIS", 1):
            modality_dir = os.path.join(split_root, modality)
            if not os.path.isdir(modality_dir):
                continue
            for filename in sorted(os.listdir(modality_dir)):
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.samples.append((os.path.join(modality_dir, filename), modality_id))
        if not self.samples:
            raise RuntimeError(f"No images found under {split_root}.")

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_degradation(self, image: Image.Image) -> Image.Image:
        """Apply random degradations: noise, blur, brightness."""
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.6, 1.4))
        if random.random() < 0.5:
            noise = torch.randn(3, image.height, image.width) * random.uniform(0.02, 0.08)
            image_tensor = BaseDataset.to_tensor(image)
            image_tensor = torch.clamp(image_tensor + noise, 0.0, 1.0)
            image = Image.fromarray((image_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        return image

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path, modality_id = self.samples[index]
        image = Image.open(path).convert("RGB")
        degraded = self._apply_degradation(image.copy())
        return {
            "input": BaseDataset.to_tensor(degraded),
            "target": BaseDataset.to_tensor(image),
            "modality": torch.tensor(modality_id, dtype=torch.long),
        }
