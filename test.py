"""Evaluation script for SMDFusion prompt-based Restormer."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from datasets.image_folder_dataset import ImageFolderDataset
from models.full_model import FullModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Restormer prompt model")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageFolderDataset(args.data_root, split="val")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = FullModel().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device)
            outputs = model(inputs)
            total += outputs.reconstructed.size(0)
    print(f"Evaluated {total} samples.")


if __name__ == "__main__":
    main()
