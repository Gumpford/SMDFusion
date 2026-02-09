"""Training entrypoint for SMDFusion prompt-based Restormer."""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from datasets.image_folder_dataset import ImageFolderDataset
from models.full_model import FullModel
from trainers.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Restormer prompt model")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda-prompt", type=float, default=0.1, help="Prompt entropy weight")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", help="Checkpoint path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageFolderDataset(args.data_root, split="train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    model = FullModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer, device, lambda_prompt=args.lambda_prompt)
    if args.resume and os.path.exists(args.checkpoint):
        trainer.load_checkpoint(args.checkpoint)
    trainer.fit(dataloader, epochs=args.epochs)
    trainer.save_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
