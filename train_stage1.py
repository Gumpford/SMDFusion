import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from models.stage1_prompted_restormer import Stage1PromptedRestormer


@dataclass
class TrainConfig:
    data_root: str
    patch_size: int = 128
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 10
    learning_rate: float = 1e-4
    seed: int = 42
    save_dir: str = "checkpoints"
    save_every: int = 1
    modality_pool_size: int = 2
    degradation_pool_size: int = 4
    prompt_dim: int = 64
    dim: int = 48


class InfraVisibleDataset(Dataset):
    def __init__(self, root: str, patch_size: int, augment: bool = True):
        self.root = Path(root)
        self.patch_size = patch_size
        self.augment = augment
        infrared_dir = self.root / "infrared"
        visible_dir = self.root / "visible"
        self.paths = []
        if infrared_dir.exists():
            self.paths.extend(sorted(infrared_dir.glob("*")))
        if visible_dir.exists():
            self.paths.extend(sorted(visible_dir.glob("*")))
        if not self.paths:
            raise FileNotFoundError("No images found in infrared/ or visible/ directories.")

    def __len__(self) -> int:
        return len(self.paths)

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("L")
        return img

    def _random_crop(self, img: Image.Image) -> Image.Image:
        if img.width < self.patch_size or img.height < self.patch_size:
            scale = max(self.patch_size / img.width, self.patch_size / img.height)
            new_size = (int(img.width * scale) + 1, int(img.height * scale) + 1)
            img = img.resize(new_size, Image.BICUBIC)
        left = random.randint(0, img.width - self.patch_size)
        top = random.randint(0, img.height - self.patch_size)
        return img.crop((left, top, left + self.patch_size, top + self.patch_size))

    def _augment(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self._load_image(self.paths[idx])
        img = self._random_crop(img)
        if self.augment:
            img = self._augment(img)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        return img


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device) - kernel_size // 2
    grid = coords[:, None] ** 2 + coords[None, :] ** 2
    kernel = torch.exp(-grid / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


def apply_gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_size = max(3, int(2 * round(sigma * 2) + 1))
    kernel = gaussian_kernel(kernel_size, sigma, x.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    x = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(x, kernel)


def apply_contrast(x: torch.Tensor, factor: float) -> torch.Tensor:
    mean = x.mean(dim=(-2, -1), keepdim=True)
    x = mean + factor * (x - mean)
    return x.clamp(0.0, 1.0)


def apply_degradation(clean: torch.Tensor) -> torch.Tensor:
    degraded = clean.clone()
    if random.random() < 0.9:
        sigma = random.uniform(0.0, 0.05)
        degraded = degraded + sigma * torch.randn_like(degraded)
    if random.random() < 0.7:
        blur_sigma = random.uniform(0.2, 1.5)
        degraded = apply_gaussian_blur(degraded, blur_sigma)
    if random.random() < 0.7:
        contrast_factor = random.uniform(0.3, 0.9)
        degraded = apply_contrast(degraded, contrast_factor)
    return degraded.clamp(0.0, 1.0)


def compute_losses(
    recon_full: torch.Tensor,
    recon_half: torch.Tensor,
    recon_quarter: torch.Tensor,
    target: torch.Tensor,
    weights: Tuple[float, float, float] = (1.0, 0.5, 0.25),
) -> Tuple[torch.Tensor, dict]:
    target_half = F.interpolate(target, scale_factor=0.5, mode="bilinear", align_corners=False)
    target_quarter = F.interpolate(target, scale_factor=0.25, mode="bilinear", align_corners=False)

    loss_full = F.l1_loss(recon_full, target)
    loss_half = F.l1_loss(recon_half, target_half)
    loss_quarter = F.l1_loss(recon_quarter, target_quarter)

    total_loss = weights[0] * loss_full + weights[1] * loss_half + weights[2] * loss_quarter
    return total_loss, {
        "loss_full": loss_full.detach(),
        "loss_half": loss_half.detach(),
        "loss_quarter": loss_quarter.detach(),
    }


def save_checkpoint(
    save_dir: str,
    epoch: int,
    encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "encoder_state": encoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(save_dir, f"stage1_epoch_{epoch}.pth"))


def load_checkpoint(path: str, encoder: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint.get("epoch", 0)


def train_stage1(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = InfraVisibleDataset(config.data_root, config.patch_size, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model = Stage1PromptedRestormer(
        in_channels=1,
        dim=config.dim,
        modality_pool_size=config.modality_pool_size,
        degradation_pool_size=config.degradation_pool_size,
        prompt_dim=config.prompt_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.epochs + 1):
        model.train()
        for step, clean in enumerate(loader, start=1):
            clean = clean.to(device)
            degraded = apply_degradation(clean)

            outputs = model(degraded)
            recon_full = outputs["recon_full"]
            recon_half = outputs["recon_half"]
            recon_quarter = outputs["recon_quarter"]

            total_loss, loss_dict = compute_losses(
                recon_full,
                recon_half,
                recon_quarter,
                clean,
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(
                    f"Epoch {epoch} Step {step} "
                    f"Loss {total_loss.item():.4f} "
                    f"Full {loss_dict['loss_full'].item():.4f} "
                    f"Half {loss_dict['loss_half'].item():.4f} "
                    f"Quarter {loss_dict['loss_quarter'].item():.4f}"
                )

        if epoch % config.save_every == 0:
            save_checkpoint(config.save_dir, epoch, model.encoder, optimizer)


def extract_features(
    encoder: nn.Module,
    image: torch.Tensor,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder.eval()
    if device is None:
        device = next(encoder.parameters()).device
    with torch.no_grad():
        image = image.to(device)
        features = encoder(image)
    return features


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Stage-1 prompted Restormer pretraining")
    parser.add_argument("--data_root", required=True, help="Dataset root with infrared/ and visible/ folders")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--modality_pool_size", type=int, default=2)
    parser.add_argument("--degradation_pool_size", type=int, default=4)
    parser.add_argument("--prompt_dim", type=int, default=64)
    parser.add_argument("--dim", type=int, default=48)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train_stage1(config)
