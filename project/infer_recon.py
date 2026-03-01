import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.recon_model import Stage1ReconModel, load_frozen_classifier
from utils.io import load_image_tensor, save_image_tensor


def parse_args():
    p = argparse.ArgumentParser("Inference for stage-1 recon model")
    p.add_argument("--image_path", type=str, default="", help="single image path")
    p.add_argument("--input_dir", type=str, default="", help="folder for batch inference")
    p.add_argument("--classifier_ckpt", type=str, required=True)
    p.add_argument("--recon_ckpt", type=str, required=True)
    p.add_argument("--dino_model_name", type=str, default="dinov2_vitb14")
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--out_ch", type=int, default=3)
    p.add_argument("--save_z", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _collect_inputs(args):
    if args.image_path:
        return [Path(args.image_path)]
    if args.input_dir:
        paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
            paths.extend(sorted(Path(args.input_dir).glob(ext)))
        return paths
    raise ValueError("Either --image_path or --input_dir must be set")


def main():
    args = parse_args()
    device = torch.device(args.device)
    in_paths = _collect_inputs(args)
    out_root = Path(args.out_path)
    out_root.mkdir(parents=True, exist_ok=True)

    classifier = load_frozen_classifier(args.classifier_ckpt, device=device)
    model = Stage1ReconModel(dino_model_name=args.dino_model_name, out_ch=3).to(device)

    ckpt = torch.load(args.recon_ckpt, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    for p in tqdm(in_paths, desc="Infer"):
        x, orig_ch = load_image_tensor(str(p), force_3ch=True)
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            cls_out = classifier(x)
            out = model(x, cls_out, enable_prompt=True)
            y = out["x_hat"][0]

        if args.out_ch == 1:
            y = y[:1]
        elif args.out_ch == 3:
            y = y[:3]
        else:
            raise ValueError("--out_ch must be 1 or 3")

        save_image_tensor(y, str(out_root / f"{p.stem}_recon.png"))
        if args.save_z:
            np.save(out_root / f"{p.stem}_z.npy", out["z"][0].detach().cpu().numpy())


if __name__ == "__main__":
    main()
