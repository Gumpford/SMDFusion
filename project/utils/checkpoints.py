from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)
