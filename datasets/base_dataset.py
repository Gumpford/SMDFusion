"""Base dataset definitions and common helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class BaseDataset(Dataset, ABC):
    """Abstract base dataset with shared utilities."""

    def __init__(self, root: str, split: str) -> None:
        super().__init__()
        self.root = root
        self.split = split

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def to_tensor(image) -> torch.Tensor:
        """Convert PIL image to float tensor in [0, 1]."""
        return TF.to_tensor(image)
