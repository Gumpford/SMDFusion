from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from dataset_clean_irvi import CleanIRVIDataset
from online_degradation import IR_DEG_NAMES, VI_DEG_NAMES, OnlineSingleDegrader
from prompt_pool import PromptPool
from utils_loss import CharbonnierLoss, EdgeLoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Stage-2 prompt training for frozen MPRNet + frozen predictor")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--mprnet_weights", type=str, default="model_denoising.pth")
    p.add_argument("--predictor_weights", type=str, required=True)
    p.add_argument("--mprnet_cls", type=str, default="MPRNet:MPRNet")
    p.add_argument("--predictor_cls", type=str, default="predictor:HierarchicalPromptPredictor")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--out_dir", type=str, default="outputs_prompt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def resolve_class(spec: str):
    module_name, class_name = spec.split(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def load_checkpoint_flexible(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "params", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {ckpt_path}. Missing={len(missing)} Unexpected={len(unexpected)}")


def freeze_model(model: torch.nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def build_amp_dtype(precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)

    mprnet = resolve_class(args.mprnet_cls)().to(device)
    predictor = resolve_class(args.predictor_cls)().to(device)

    load_checkpoint_flexible(mprnet, args.mprnet_weights)
    load_checkpoint_flexible(predictor, args.predictor_weights)

    freeze_model(mprnet)
    freeze_model(predictor)

    prompt_pool = PromptPool(num_prompts=12, channels=3, base_hw=(256, 256)).to(device)

    optimizer = torch.optim.AdamW(prompt_pool.parameters(), lr=args.lr, weight_decay=0.0)

    char_loss = CharbonnierLoss().to(device)
    edge_loss = EdgeLoss().to(device)
    degrader = OnlineSingleDegrader()

    ds = CleanIRVIDataset(root=args.data_root, patch_size=args.patch_size, random_crop=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    amp_dtype = build_amp_dtype(args.precision)
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    for epoch in range(1, args.epochs + 1):
        prompt_pool.train()
        epoch_loss = 0.0
        epoch_mpr = 0.0
        epoch_reg = 0.0

        for batch in loader:
            target = batch["clean"].to(device, non_blocking=True)
            modalities = batch["modality"]

            degraded = []
            for i in range(target.size(0)):
                dimg, _ = degrader.degrade(target[i : i + 1], modalities[i])
                degraded.append(dimg)
            degraded = torch.cat(degraded, dim=0)

            x_pred = F.interpolate(degraded, size=(224, 224), mode="bilinear", align_corners=False)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = predictor(images=x_pred)
                p = out["P"]  # (B,12), already modality gated

                prompt = prompt_pool.compose(p, out_hw=degraded.shape[-2:])
                input_p = torch.clamp(degraded + prompt, 0.0, 1.0)

                restored_list = mprnet(input_p)
                if not isinstance(restored_list, (list, tuple)) or len(restored_list) != 3:
                    raise RuntimeError("MPRNet forward must return [stage3, stage2, stage1]")

                sum_char = sum(char_loss(s, target) for s in restored_list)
                sum_edge = sum(edge_loss(s, target) for s in restored_list)
                loss_mpr = sum_char + 0.05 * sum_edge

                loss_reg = torch.mean(prompt * prompt)
                loss_total = loss_mpr + 1e-4 * loss_reg

            if scaler.is_enabled():
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                optimizer.step()

            epoch_loss += loss_total.item()
            epoch_mpr += loss_mpr.item()
            epoch_reg += loss_reg.item()

        n = len(loader)
        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"loss={epoch_loss/n:.6f} mpr={epoch_mpr/n:.6f} reg={epoch_reg/n:.6f}"
        )

        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            ckpt_path = Path(args.out_dir) / f"prompt_epoch_{epoch:03d}.pth"
            torch.save(
                {
                    "prompt_bank": prompt_pool.prompt_bank.detach().cpu(),
                    "ir_names": IR_DEG_NAMES,
                    "vi_names": VI_DEG_NAMES,
                    "base_hw": (256, 256),
                },
                ckpt_path,
            )
            print(f"Saved prompt checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
