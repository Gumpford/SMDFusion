import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
import torch
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
from tqdm import tqdm

from data.dataset import MixedCleanDataset
from degradations.ir_degradations import sample_ir_degradation
from degradations.vi_degradations import sample_vi_degradation
from losses.ssim import ssim
from losses.edge import sobel_edge_loss
from models.recon_model import ReconModel
from utils.seed import set_seed
from utils.meters import AverageMeter
from utils.checkpoints import save_checkpoint


def degrade_batch(x_clean, modality):
    xs = []
    for x, m in zip(x_clean, modality):
        xs.append(sample_ir_degradation(x) if int(m.item()) == 0 else sample_vi_degradation(x))
    return torch.stack(xs, dim=0)


def run_epoch(model, loader, optimizer, scaler, device, args, step_state, train=True):
    model.train(train)
    meter = AverageMeter()
    for x_clean, modality, _ in tqdm(loader, disable=not train):
        x_clean = x_clean.to(device)
        modality = modality.to(device)
        x_deg = degrade_batch(x_clean.cpu(), modality.cpu()).to(device)

        in_phase2 = step_state["step"] >= step_state["warmup_steps"]
        model.set_prompt_trainable(in_phase2)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp and device.startswith("cuda")):
            x_hat, z, gates, _ = model(x_deg, enable_prompt=in_phase2, temperature=args.prompt_temp)
            l1 = (x_hat - x_clean).abs().mean()
            l_ssim = 1 - ssim(x_hat, x_clean)
            edge = sobel_edge_loss(x_hat, x_clean)
            orth = model.prompt_pool.orth_loss() if in_phase2 else x_hat.new_tensor(0.0)
            gate = torch.stack([g.abs().mean() for g in gates]).mean() if in_phase2 else x_hat.new_tensor(0.0)
            loss = l1 + 0.5 * l_ssim + 0.1 * edge + args.lambda_orth * orth + args.lambda_gate * gate

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step_state["step"] += 1
            if step_state["step"] % args.log_interval == 0:
                print(f"step {step_state['step']} total={loss.item():.4f} l1={l1.item():.4f} ssim={l_ssim.item():.4f} edge={edge.item():.4f} orth={orth.item():.4f} gate={gate.item():.4f}")
        meter.update(loss.item(), x_clean.size(0))
    return meter.avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--classifier_ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dino_model_name", default="dinov2_vitb14")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patch_size", type=int, default=224)
    ap.add_argument("--lambda_orth", type=float, default=1e-4)
    ap.add_argument("--lambda_gate", type=float, default=1e-5)
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--save_images_interval", type=int, default=1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--prompt_temp", type=float, default=1.0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.save_dir) / "vis").mkdir(exist_ok=True)

    ds = MixedCleanDataset(args.dataset_root, patch_size=args.patch_size)
    n_val = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    va_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ReconModel(args.classifier_ckpt, args.dino_model_name, hidden_dim=256).to(device)
    model.freeze_backbones()
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler(enabled=args.amp and device.startswith("cuda"))

    total_steps = args.epochs * len(tr_loader)
    step_state = {"step": 0, "warmup_steps": int(0.2 * total_steps)}
    best = 1e9

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, tr_loader, optim, scaler, device, args, step_state, train=True)
        with torch.no_grad():
            va = run_epoch(model, va_loader, optim, scaler, device, args, step_state, train=False)
        print(f"Epoch {epoch}/{args.epochs} train={tr:.4f} val={va:.4f}")

        save_checkpoint({"model": model.state_dict(), "epoch": epoch, "val": va}, f"{args.save_dir}/last.pth")
        prompt_only = {
            "prompt_pool": model.prompt_pool.state_dict(),
            "adapters": model.adapters.state_dict(),
            "proj": model.encoder.proj.state_dict(),
        }
        save_checkpoint(prompt_only, f"{args.save_dir}/prompt_only.pth")
        if va < best:
            best = va
            save_checkpoint({"model": model.state_dict(), "epoch": epoch, "val": va}, f"{args.save_dir}/best.pth")

        if epoch % args.save_images_interval == 0:
            x_clean, mod, _ = next(iter(va_loader))
            x_clean = x_clean.to(device)
            x_deg = degrade_batch(x_clean.cpu(), mod).to(device)
            x_hat, *_ = model(x_deg, enable_prompt=True)
            vis = torch.cat([x_deg[:4], x_hat[:4].clamp(0, 1), x_clean[:4]], dim=0)
            save_image(vis, f"{args.save_dir}/vis/epoch_{epoch:03d}.png", nrow=4)


if __name__ == "__main__":
    main()
