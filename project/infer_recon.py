import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.recon_model import Stage1ReconModel
from utils.io import load_image_tensor, save_image_tensor


def parse_args():
    p = argparse.ArgumentParser("Infer stage-1 recon")
    p.add_argument("--image_path", type=str, default="")
    p.add_argument("--input_dir", type=str, default="")
    p.add_argument("--classifier_ckpt", type=str, required=True)
    p.add_argument("--recon_ckpt", type=str, required=True)
    p.add_argument("--dino_model_name", type=str, default="dinov2_vitb14")
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--out_ch", type=int, default=3, choices=[1, 3])
    p.add_argument("--save_z", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def gather_inputs(image_path: str, input_dir: str):
    if image_path:
        return [Path(image_path)]
    if input_dir:
        paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
            paths.extend(sorted(Path(input_dir).glob(ext)))
        return paths
    raise ValueError("Set --image_path or --input_dir")


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = Stage1ReconModel(
        classifier_ckpt=args.classifier_ckpt,
        device=device,
        dino_model_name=args.dino_model_name,
        hidden_dim=256,
    ).to(device)
    ckpt = torch.load(args.recon_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=False)
    model.eval()

    paths = gather_inputs(args.image_path, args.input_dir)
    out_dir = Path(args.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(paths, desc="infer"):
        x, _ = load_image_tensor(str(p), force_3ch=True)
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x, enable_prompt=True)
            y = out["x_hat"][0]

        if args.out_ch == 1:
            y = y[:1]
        save_image_tensor(y, str(out_dir / f"{p.stem}_recon.png"))
        if args.save_z:
            np.save(out_dir / f"{p.stem}_z.npy", out["z"][0].detach().cpu().numpy())


if __name__ == "__main__":
    main()
