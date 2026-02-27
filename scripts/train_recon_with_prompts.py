"""Train prompt-injected reconstruction with frozen predictor + frozen DINOv2 encoder.

Example:
python scripts/train_recon_with_prompts.py ^
  --data_root D:\DataSet\TrainData\MSRS\msrs4_paired ^
  --save_dir D:\exp\recon_prompts ^
  --predictor_ckpt D:\exp\predictor\predictor_best.pth ^
  --Kir 6 --Kvi 6 --D 256 ^
  --dinov2_name dinov2_vits14 ^
  --batch_size 8 --epochs 50 --lr 1e-4 ^
  --lambda_ssim 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.recon_dataset import PairedReconDataset
from losses.ssim_loss import ssim
from models.dinov2_autoencoder import DINOv2AutoEncoder
from predictors.hierarchical_predictor import HierarchicalPromptPredictor, PredictorConfig
from prompts.prompt_bank import PromptBank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train reconstruction with predictor-driven prompts")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--predictor_ckpt", type=str, required=True)
    p.add_argument("--dinov2_name", type=str, default="dinov2_vits14")
    p.add_argument("--dinov2_local_ckpt", type=str, default="")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_ssim", type=float, default=1.0)
    p.add_argument("--Kir", type=int, required=True)
    p.add_argument("--Kvi", type=int, required=True)
    p.add_argument("--D", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", type=str, default="none", choices=["fp16", "bf16", "none"])
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--crop_size", type=int, default=256)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _amp_dtype(mode: str) -> torch.dtype | None:
    if mode == "fp16":
        return torch.float16
    if mode == "bf16":
        return torch.bfloat16
    return None


def build_predictor_from_ckpt(ckpt_path: str, kir: int, kvi: int, device: torch.device) -> HierarchicalPromptPredictor:
    payload = torch.load(ckpt_path, map_location=device)
    args = payload.get("args", {})
    cfg = PredictorConfig(
        feat_dim=args.get("feat_dim", 512),
        kir=kir,
        kvi=kvi,
        hidden_dim=args.get("hidden_dim", 512),
        temperature=args.get("temperature", 1.0),
        label_smoothing=args.get("label_smoothing", 0.0),
        use_clip=args.get("use_clip", False),
        clip_model_name=args.get("clip_model_name", "ViT-H-14"),
        freeze_clip=True,
    )
    model = HierarchicalPromptPredictor(cfg).to(device)
    state = payload["model"] if "model" in payload else payload.get("predictor", payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def run_val(
    predictor: HierarchicalPromptPredictor,
    prompt_bank: PromptBank,
    ae: DINOv2AutoEncoder,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    predictor.eval()
    ae.eval()
    prompt_bank.eval()

    n = 0
    mse_sum = 0.0
    ssim_sum = 0.0
    for batch in loader:
        x_deg = batch["x_deg"].to(device)
        x_gt = batch["x_gt"].to(device)
        b = x_deg.size(0)

        assert x_deg.shape == x_gt.shape and x_deg.ndim == 4 and x_deg.size(1) == 1

        p = predictor.predict_P(x_deg)
        prompt_vec = prompt_bank(p)
        y_hat = ae(x_deg, prompt_vec)

        mse_v = F.mse_loss(y_hat, x_gt)
        ssim_v = ssim(y_hat, x_gt)
        mse_sum += mse_v.item() * b
        ssim_sum += ssim_v.item() * b
        n += b

    return {
        "mse": mse_sum / max(1, n),
        "ssim": ssim_sum / max(1, n),
        "loss": (mse_sum / max(1, n)) + (1.0 - ssim_sum / max(1, n)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = args.Kir + args.Kvi

    train_ds = PairedReconDataset(args.data_root, split="train", crop_size=args.crop_size, train=True)
    val_ds = PairedReconDataset(args.data_root, split="val", crop_size=args.crop_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    predictor = build_predictor_from_ckpt(args.predictor_ckpt, args.Kir, args.Kvi, device)
    prompt_bank = PromptBank(k=k, d=args.D).to(device)
    ae = DINOv2AutoEncoder(prompt_dim=args.D, dinov2_name=args.dinov2_name, dinov2_local_ckpt=args.dinov2_local_ckpt).to(device)
    ae.freeze_encoder()  # encoder must stay frozen during reconstruction training.

    for p in ae.encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(list(prompt_bank.parameters()) + list(ae.trainable_parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.1)

    amp_dtype = _amp_dtype(args.amp)
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp == "fp16" and device.type == "cuda"))

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        prompt_bank.load_state_dict(ckpt["prompt_bank"])
        ae.load_state_dict(ckpt["autoencoder"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", best_loss)

    label_map = {"Kir": args.Kir, "Kvi": args.Kvi, "K": k}
    with open(save_dir / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    for epoch in range(start_epoch, args.epochs):
        predictor.eval()
        ae.train()
        prompt_bank.train()

        epoch_mse = 0.0
        epoch_ssim = 0.0
        epoch_loss = 0.0
        n = 0

        for batch in train_loader:
            x_deg = batch["x_deg"].to(device)
            x_gt = batch["x_gt"].to(device)
            b = x_deg.size(0)

            assert x_deg.shape == x_gt.shape and x_deg.ndim == 4 and x_deg.size(1) == 1

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                p = predictor.predict_P(x_deg)
            assert p.shape == (b, k)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                prompt_vec = prompt_bank(p)
                assert prompt_vec.shape == (b, args.D)
                y_hat = ae(x_deg, prompt_vec)
                mse_v = F.mse_loss(y_hat, x_gt)
                ssim_v = ssim(y_hat, x_gt)
                recon_loss = mse_v + args.lambda_ssim * (1.0 - ssim_v)
                reg_loss = 1e-4 * torch.norm(prompt_bank.bank, p=2)
                loss = recon_loss + reg_loss

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            global_step += 1
            epoch_mse += mse_v.item() * b
            epoch_ssim += ssim_v.item() * b
            epoch_loss += loss.item() * b
            n += b

        scheduler.step()

        train_mse = epoch_mse / max(1, n)
        train_ssim = epoch_ssim / max(1, n)
        train_loss = epoch_loss / max(1, n)
        val_metrics = run_val(predictor, prompt_bank, ae, val_loader, device)

        print(
            f"epoch={epoch+1}/{args.epochs} "
            f"train_loss={train_loss:.6f} train_mse={train_mse:.6f} train_ssim={train_ssim:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_mse={val_metrics['mse']:.6f} val_ssim={val_metrics['ssim']:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "args": vars(args),
            "predictor_config": {
                "Kir": args.Kir,
                "Kvi": args.Kvi,
                "predictor_ckpt": args.predictor_ckpt,
            },
            "prompt_bank": prompt_bank.state_dict(),
            "autoencoder": ae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        if scaler.is_enabled():
            ckpt["scaler"] = scaler.state_dict()

        torch.save(ckpt, save_dir / "latest.pth")
        torch.save(prompt_bank.state_dict(), save_dir / "prompt_bank_only.pth")

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            ckpt["best_loss"] = best_loss
            torch.save(ckpt, save_dir / "best.pth")


if __name__ == "__main__":
    main()
