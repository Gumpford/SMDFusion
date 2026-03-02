import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
import numpy as np
import torch

from models.recon_model import ReconModel
from utils.io import load_image, save_tensor_image, list_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True, help="single image or folder")
    ap.add_argument("--classifier_ckpt", required=True)
    ap.add_argument("--recon_ckpt", required=True)
    ap.add_argument("--dino_model_name", default="dinov2_vitb14")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--out_ch", type=int, default=3, choices=[1, 3])
    ap.add_argument("--save_z", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = ReconModel(args.classifier_ckpt, args.dino_model_name, hidden_dim=256).to(device)
    ckpt = torch.load(args.recon_ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    images = list_images(args.image_path)
    out_root = Path(args.out_path)
    out_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for p in images:
            x = load_image(str(p)).unsqueeze(0).to(device)
            x_hat, z, _, _ = model(x, enable_prompt=True)
            out_file = out_root / f"{p.stem}_recon.png"
            save_tensor_image(x_hat, str(out_file), out_ch=args.out_ch)
            if args.save_z:
                np.save(out_root / f"{p.stem}_z.npy", z.squeeze(0).cpu().numpy())
            print(f"saved: {out_file}")


if __name__ == "__main__":
    main()
