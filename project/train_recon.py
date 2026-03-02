import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CleanReconDataset, split_items
from degradations.ir_degradations import apply_ir_degradation, sample_ir_degradations
from degradations.vi_degradations import apply_vi_degradation, sample_vi_degradations
from losses.edge import SobelEdgeLoss
from losses.ssim import SSIMLoss
from models.recon_model import Stage1ReconModel
from utils.checkpoints import save_checkpoint
from utils.io import save_image_tensor
from utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser("Train Stage-1 recon")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--val_root", type=str, default="")
    p.add_argument("--classifier_ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dino_model_name", type=str, default="dinov2_vitb14")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch_size", type=int, default=224)
    p.add_argument("--lambda_orth", type=float, default=1e-3)
    p.add_argument("--lambda_gate", type=float, default=1e-5)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_images_interval", type=int, default=1)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def build_loaders(args):
    base = CleanReconDataset(args.dataset_root, patch_size=args.patch_size)
    if args.val_root:
        tr_ds = base
        va_ds = CleanReconDataset(args.val_root, patch_size=args.patch_size)
    else:
        tr_items, va_items = split_items(base.items, args.val_ratio, args.seed)
        tr_ds = CleanReconDataset(args.dataset_root, items=tr_items, patch_size=args.patch_size)
        va_ds = CleanReconDataset(args.dataset_root, items=va_items, patch_size=args.patch_size)

    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return tr, va


def make_degraded_batch(x_clean: torch.Tensor, modality_gt: torch.Tensor, seed_base: int) -> torch.Tensor:
    """Apply online modality-aware degradations on CPU numpy and return torch tensor."""
    b, c, h, w = x_clean.shape
    out = []
    for i in range(b):
        rng = np.random.RandomState(seed_base + i)
        x = x_clean[i].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        if int(modality_gt[i].item()) == 0:
            deg_names = sample_ir_degradations(rng)
            y = x.copy()
            for n in deg_names:
                y = apply_ir_degradation(y, n, rng)
        else:
            deg_names = sample_vi_degradations(rng)
            y = x.copy()
            for n in deg_names:
                y = apply_vi_degradation(y, n, rng)
        out.append(torch.from_numpy(y).permute(2, 0, 1))
    return torch.stack(out, dim=0).to(x_clean.device, dtype=x_clean.dtype)


def set_phase_trainable(model: Stage1ReconModel, warmup: bool):
    for p in model.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = False
    for p in model.encoder.dino.parameters():
        p.requires_grad = False
    if warmup:
        for p in model.prompt_pool.parameters():
            p.requires_grad = False
        for p in model.adapters.parameters():
            p.requires_grad = False


def run_epoch(loader, model, optimizer, scaler, args, device, epoch: int, warmup_steps: int, train: bool):
    model.train(train)
    if not train:
        model.eval()

    ssim_fn = SSIMLoss().to(device)
    edge_fn = SobelEdgeLoss().to(device)
    total_loss = 0.0
    n_samples = 0

    pbar = tqdm(enumerate(loader), total=len(loader), desc=("Train" if train else "Val") + f" {epoch}")
    for it, batch in pbar:
        x_clean = batch["x_clean"].to(device)
        modality_gt = batch["modality_gt"].to(device)
        step = epoch * len(loader) + it
        enable_prompt = step >= warmup_steps
        if train and it == 0:
            set_phase_trainable(model, warmup=not enable_prompt)

        x_deg = make_degraded_batch(x_clean, modality_gt, seed_base=args.seed + step * 1000)

        amp_on = device.type == "cuda" and (not args.no_amp)
        with autocast(enabled=amp_on):
            out = model(x_deg, enable_prompt=enable_prompt)
            x_hat = out["x_hat"]
            l1 = F.l1_loss(x_hat, x_clean)
            ssim = ssim_fn(x_hat, x_clean)
            edge = edge_fn(x_hat, x_clean)
            orth = model.prompt_pool.orth_loss() if enable_prompt else torch.tensor(0.0, device=device)
            gate = torch.stack([g.abs().mean() for g in out["gates"]]).mean() if enable_prompt else torch.tensor(0.0, device=device)
            loss = l1 + 0.5 * ssim + 0.1 * edge + args.lambda_orth * orth + args.lambda_gate * gate

        if train:
            optimizer.zero_grad(set_to_none=True)
            if amp_on:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * x_clean.size(0)
        n_samples += x_clean.size(0)

        if (it + 1) % args.log_interval == 0 or (it + 1) == len(loader):
            pbar.set_postfix({
                "L1": f"{l1.item():.4f}",
                "SSIM": f"{ssim.item():.4f}",
                "edge": f"{edge.item():.4f}",
                "orth": f"{orth.item():.4f}",
                "gate": f"{gate.item():.4f}",
                "total": f"{loss.item():.4f}",
            })

        if train and it == 0 and (epoch % args.save_images_interval == 0):
            vis = Path(args.save_dir) / "vis"
            vis.mkdir(parents=True, exist_ok=True)
            save_image_tensor(x_deg[0], str(vis / f"ep{epoch:03d}_deg.png"))
            save_image_tensor(x_hat[0], str(vis / f"ep{epoch:03d}_hat.png"))
            save_image_tensor(x_clean[0], str(vis / f"ep{epoch:03d}_gt.png"))

    return total_loss / max(1, n_samples)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    tr_loader, va_loader = build_loaders(args)
    model = Stage1ReconModel(
        classifier_ckpt=args.classifier_ckpt,
        device=device,
        dino_model_name=args.dino_model_name,
        hidden_dim=256,
    ).to(device)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler(enabled=(device.type == "cuda" and (not args.no_amp)))

    total_steps = args.epochs * len(tr_loader)
    warmup_steps = int(total_steps * 0.2)

    best_val = 1e9
    for ep in range(args.epochs):
        tr_loss = run_epoch(tr_loader, model, optim, scaler, args, device, ep, warmup_steps, train=True)
        with torch.no_grad():
            va_loss = run_epoch(va_loader, model, optim, scaler, args, device, ep, warmup_steps, train=False)

        state = {
            "epoch": ep,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "args": vars(args),
        }
        save_checkpoint(state, str(Path(args.save_dir) / "last.pth"))
        save_checkpoint(model.prompt_state_dict(), str(Path(args.save_dir) / "prompt_only.pth"))
        if va_loss < best_val:
            best_val = va_loss
            save_checkpoint(state, str(Path(args.save_dir) / "best.pth"))

        print(f"[Epoch {ep}] train={tr_loss:.4f}, val={va_loss:.4f}, best={best_val:.4f}")


if __name__ == "__main__":
    main()
