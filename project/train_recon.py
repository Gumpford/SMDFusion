import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CleanToDegradedReconDataset, split_items
from losses.edge import SobelEdgeLoss
from losses.ssim import SSIMLoss
from models.recon_model import Stage1ReconModel, load_frozen_classifier
from utils.checkpoints import save_checkpoint
from utils.io import save_image_tensor
from utils.meters import AverageMeter
from utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser("Stage-1 transferable prompt training")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--classifier_ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dino_model_name", type=str, default="dinov2_vitb14")
    p.add_argument("--out_ch", type=int, default=3)
    p.add_argument("--save_dir", type=str, default="runs/stage1_recon")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--val_root", type=str, default="")
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--print_freq", type=int, default=50)
    p.add_argument("--lambda_orth", type=float, default=1e-3)
    p.add_argument("--lambda_gate", type=float, default=1e-5)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def build_loaders(args):
    full_ds = CleanToDegradedReconDataset(args.dataset_root, crop_size=args.crop_size, seed=args.seed, force_3ch=True)
    if args.val_root:
        train_ds = full_ds
        val_ds = CleanToDegradedReconDataset(args.val_root, crop_size=args.crop_size, seed=args.seed + 999, force_3ch=True)
    else:
        tr_items, va_items = split_items(full_ds.items, val_ratio=args.val_ratio, seed=args.seed)
        train_ds = CleanToDegradedReconDataset(args.dataset_root, items=tr_items, crop_size=args.crop_size, seed=args.seed, force_3ch=True)
        val_ds = CleanToDegradedReconDataset(args.dataset_root, items=va_items, crop_size=args.crop_size, seed=args.seed + 999, force_3ch=True)

    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    va_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return tr_loader, va_loader


def set_trainable_for_phase(model: Stage1ReconModel, warmup: bool):
    for p in model.parameters():
        p.requires_grad = True
    # frozen DINO backbone
    for p in model.encoder.dino.parameters():
        p.requires_grad = False
    if warmup:
        for p in model.prompt_pool.parameters():
            p.requires_grad = False
        for p in model.adapters.parameters():
            p.requires_grad = False


def compute_losses(model_out: Dict[str, torch.Tensor], x_clean: torch.Tensor, ssim_loss_fn, edge_loss_fn, lambda_orth, lambda_gate, enable_prompt):
    x_hat = model_out["x_hat"]
    l1 = F.l1_loss(x_hat, x_clean)
    ssim = ssim_loss_fn(x_hat, x_clean)
    edge = edge_loss_fn(x_hat, x_clean)
    orth = model_out.get("orth_loss", torch.tensor(0.0, device=x_hat.device))
    gate_l1 = torch.stack([g.abs().mean() for g in model_out["gates"]]).mean() if enable_prompt else torch.tensor(0.0, device=x_hat.device)
    total = l1 + 0.5 * ssim + 0.1 * edge + lambda_orth * orth + lambda_gate * gate_l1
    return {"total": total, "l1": l1, "ssim": ssim, "edge": edge, "orth": orth, "gate_l1": gate_l1}


def run_epoch(loader, model, classifier, optimizer, scaler, device, epoch, total_steps, warmup_steps, args, train=True):
    model.train(train)
    if not train:
        model.eval()

    meter = AverageMeter()
    ssim_loss_fn = SSIMLoss().to(device)
    edge_loss_fn = SobelEdgeLoss().to(device)

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"{'Train' if train else 'Val'} {epoch}")
    for i, batch in pbar:
        x_clean = batch["x_clean"].to(device)
        x_deg = batch["x_deg"].to(device)

        global_step = epoch * len(loader) + i
        enable_prompt = global_step >= warmup_steps

        if train and i == 0:
            set_trainable_for_phase(model, warmup=not enable_prompt)

        with torch.no_grad():
            cls_outputs = classifier(x_deg)
            required = ["p_mod", "p_deg_ir", "p_deg_vi"]
            for k in required:
                if k not in cls_outputs:
                    raise KeyError(f"Classifier output missing key={k}")

        amp_on = (device.type == "cuda") and (not args.no_amp)
        with autocast(enabled=amp_on):
            out = model(x_deg, cls_outputs, enable_prompt=enable_prompt)
            out["orth_loss"] = model.prompt_pool.orth_loss() if enable_prompt else torch.tensor(0.0, device=device)
            losses = compute_losses(out, x_clean, ssim_loss_fn, edge_loss_fn, args.lambda_orth, args.lambda_gate, enable_prompt)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if amp_on:
                scaler.scale(losses["total"]).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                optimizer.step()

        meter.update(losses["total"].item(), n=x_clean.size(0))
        if (i + 1) % args.print_freq == 0 or (i + 1) == len(loader):
            pbar.set_postfix({k: f"{v.item():.4f}" for k, v in losses.items()})

        if train and i == 0:
            vis_dir = Path(args.save_dir) / "vis"
            vis_dir.mkdir(parents=True, exist_ok=True)
            save_image_tensor(x_deg[0].detach().cpu(), str(vis_dir / f"ep{epoch:03d}_deg.png"))
            save_image_tensor(out["x_hat"][0].detach().cpu(), str(vis_dir / f"ep{epoch:03d}_hat.png"))
            save_image_tensor(x_clean[0].detach().cpu(), str(vis_dir / f"ep{epoch:03d}_gt.png"))

    return meter.avg


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(args)
    model = Stage1ReconModel(dino_model_name=args.dino_model_name, out_ch=args.out_ch).to(device)
    classifier = load_frozen_classifier(args.classifier_ckpt, device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(0.2 * total_steps)

    best_val = 1e9
    for epoch in range(args.epochs):
        train_loss = run_epoch(train_loader, model, classifier, optimizer, scaler, device, epoch, total_steps, warmup_steps, args, train=True)
        with torch.no_grad():
            val_loss = run_epoch(val_loader, model, classifier, optimizer, scaler, device, epoch, total_steps, warmup_steps, args, train=False)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
        }
        save_checkpoint(ckpt, str(Path(args.save_dir) / "last.pth"))
        save_checkpoint(model.prompt_state_dict(), str(Path(args.save_dir) / "prompt_only.pth"))

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt, str(Path(args.save_dir) / "best.pth"))

        print(f"[Epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f} best={best_val:.4f}")


if __name__ == "__main__":
    main()
