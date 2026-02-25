"""Train hierarchical prompt gating predictor.

Example:
python scripts/train_predictor.py --data_root /path/to/data --batch_size 16 --epochs 20 --lr 1e-4 --save_dir runs/hier_pred
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torchvision import transforms
except ImportError as exc:
    raise ImportError("torchvision is required for train_predictor.py") from exc

from data.hier_dataset import HierarchicalDegradationDataset
from losses.focal_loss import MultiClassFocalLoss
from predictors.hierarchical_predictor import HierarchicalPromptPredictor, PredictorConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hierarchical prompt predictor")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_clip", action="store_true")
    p.add_argument("--clip_model_name", type=str, default="ViT-H-14")
    p.add_argument("--freeze_clip", action="store_true")
    p.add_argument("--precision", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.2)
    p.add_argument("--routing_mode", type=str, default="gt", choices=["gt", "soft"])
    p.add_argument("--save_dir", type=str, default="checkpoints/predictor")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--feat_dim", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    return p.parse_args()


def _autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def compute_deg_loss(
    out: dict[str, torch.Tensor],
    y_mod: torch.Tensor,
    y_deg: torch.Tensor,
    focal_ir: nn.Module,
    focal_vi: nn.Module,
    use_soft_routing: bool,
) -> torch.Tensor:
    logits_ir = out["logits_deg_ir"]
    logits_vi = out["logits_deg_vi"]
    p_mod = out["p_mod"]

    mask_ir = y_mod == 0
    mask_vi = y_mod == 1

    loss = logits_ir.new_zeros(())
    cnt = 0

    if mask_ir.any():
        l_ir = focal_ir(logits_ir[mask_ir], y_deg[mask_ir])
        if use_soft_routing:
            l_ir = l_ir * p_mod[mask_ir, 0].mean()
        loss = loss + l_ir
        cnt += 1

    if mask_vi.any():
        l_vi = focal_vi(logits_vi[mask_vi], y_deg[mask_vi])
        if use_soft_routing:
            l_vi = l_vi * p_mod[mask_vi, 1].mean()
        loss = loss + l_vi
        cnt += 1

    if cnt == 0:
        return logits_ir.new_zeros(())
    return loss / cnt


@torch.no_grad()
def evaluate(
    model: HierarchicalPromptPredictor,
    loader: DataLoader,
    ce_mod: nn.Module,
    focal_ir: nn.Module,
    focal_vi: nn.Module,
    device: torch.device,
    beta: float,
    precision: str,
) -> dict[str, float]:
    model.eval()
    dtype = _autocast_dtype(precision)
    use_amp = dtype is not None and device.type == "cuda"

    total = n = 0
    total_mod = total_deg = 0.0
    mod_correct = 0
    ir_correct = ir_total = 0
    vi_correct = vi_total = 0

    for batch in loader:
        x = batch["image"].to(device)
        y_mod = batch["y_mod"].to(device)
        y_deg = batch["y_deg"].to(device)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            out = model(images=x)
            l_mod = ce_mod(out["logits_mod"], y_mod)
            l_deg = compute_deg_loss(out, y_mod, y_deg, focal_ir, focal_vi, use_soft_routing=True)
            l_total = l_mod + beta * l_deg

        total_mod += l_mod.item() * x.size(0)
        total_deg += l_deg.item() * x.size(0)
        total += l_total.item() * x.size(0)
        n += x.size(0)

        pred_mod = out["logits_mod"].argmax(dim=1)
        mod_correct += (pred_mod == y_mod).sum().item()

        mask_ir = y_mod == 0
        if mask_ir.any():
            pred_ir = out["logits_deg_ir"][mask_ir].argmax(dim=1)
            ir_correct += (pred_ir == y_deg[mask_ir]).sum().item()
            ir_total += mask_ir.sum().item()

        mask_vi = y_mod == 1
        if mask_vi.any():
            pred_vi = out["logits_deg_vi"][mask_vi].argmax(dim=1)
            vi_correct += (pred_vi == y_deg[mask_vi]).sum().item()
            vi_total += mask_vi.sum().item()

    return {
        "loss_total": total / max(n, 1),
        "loss_mod": total_mod / max(n, 1),
        "loss_deg": total_deg / max(n, 1),
        "acc_mod": mod_correct / max(n, 1),
        "acc_deg_ir": ir_correct / max(ir_total, 1),
        "acc_deg_vi": vi_correct / max(vi_total, 1),
    }


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    train_ds = HierarchicalDegradationDataset(args.data_root, split="train", transform=transform)
    val_ds = HierarchicalDegradationDataset(args.data_root, split="val", transform=transform)

    with open(save_dir / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(train_ds.get_label_maps(), f, ensure_ascii=False, indent=2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    cfg = PredictorConfig(
        feat_dim=args.feat_dim,
        kir=train_ds.Kir,
        kvi=train_ds.Kvi,
        use_clip=args.use_clip,
        clip_model_name=args.clip_model_name,
        freeze_clip=args.freeze_clip,
        temperature=args.temperature,
        label_smoothing=args.label_smoothing,
    )
    model = HierarchicalPromptPredictor(cfg).to(device)

    ce_mod = nn.CrossEntropyLoss()
    focal_ir = MultiClassFocalLoss(gamma=args.focal_gamma)
    focal_vi = MultiClassFocalLoss(gamma=args.focal_gamma)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remain = max(total_steps - warmup_steps, 1)
        return max(0.0, (total_steps - step) / remain)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == "fp16" and device.type == "cuda"))

    start_epoch, global_step, best_score = 0, 0, -1e9
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_score = ckpt.get("best_score", best_score)
        print(f"Resumed from {args.resume} at epoch={start_epoch}, global_step={global_step}")

    dtype = _autocast_dtype(args.precision)
    use_amp = dtype is not None and device.type == "cuda"

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for it, batch in enumerate(train_loader):
            x = batch["image"].to(device)
            y_mod = batch["y_mod"].to(device)
            y_deg = batch["y_deg"].to(device)

            optimizer.zero_grad(set_to_none=True)
            use_soft = args.routing_mode == "soft" or global_step >= warmup_steps

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                out = model(images=x)
                l_mod = ce_mod(out["logits_mod"], y_mod)
                l_deg = compute_deg_loss(out, y_mod, y_deg, focal_ir, focal_vi, use_soft_routing=use_soft)
                l_total = l_mod + args.beta * l_deg

            if scaler.is_enabled():
                scaler.scale(l_total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                l_total.backward()
                optimizer.step()
            scheduler.step()
            global_step += 1

            with torch.no_grad():
                pred_mod = out["logits_mod"].argmax(dim=1)
                acc_mod = (pred_mod == y_mod).float().mean().item()
                mask_ir = y_mod == 0
                mask_vi = y_mod == 1
                acc_deg_ir = (
                    (out["logits_deg_ir"][mask_ir].argmax(dim=1) == y_deg[mask_ir]).float().mean().item()
                    if mask_ir.any()
                    else 0.0
                )
                acc_deg_vi = (
                    (out["logits_deg_vi"][mask_vi].argmax(dim=1) == y_deg[mask_vi]).float().mean().item()
                    if mask_vi.any()
                    else 0.0
                )

            print(
                f"epoch={epoch+1}/{args.epochs} iter={it+1}/{len(train_loader)} lr={scheduler.get_last_lr()[0]:.6e} "
                f"L_mod={l_mod.item():.4f} L_deg={l_deg.item():.4f} L_total={l_total.item():.4f} "
                f"acc_mod={acc_mod:.4f} acc_deg_ir={acc_deg_ir:.4f} acc_deg_vi={acc_deg_vi:.4f}"
            )

        metrics = evaluate(model, val_loader, ce_mod, focal_ir, focal_vi, device, args.beta, args.precision)
        score = metrics["acc_mod"] + 0.5 * (metrics["acc_deg_ir"] + metrics["acc_deg_vi"])
        print(
            "[VAL] "
            + " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            + f" score={score:.4f}"
        )

        latest_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_score": best_score,
            "args": vars(args),
            "label_maps": train_ds.get_label_maps(),
        }
        if scaler.is_enabled():
            latest_payload["scaler"] = scaler.state_dict()
        torch.save(latest_payload, save_dir / "predictor_latest.pth")

        if score > best_score:
            best_score = score
            latest_payload["best_score"] = best_score
            torch.save(latest_payload, save_dir / "predictor_best.pth")


if __name__ == "__main__":
    main()
