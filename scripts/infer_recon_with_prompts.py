"""Inference for prompt-injected reconstruction (paired IR/VI inputs)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from models.dinov2_autoencoder import DINOv2AutoEncoder
from predictors.hierarchical_predictor import HierarchicalPromptPredictor, PredictorConfig
from prompts.prompt_bank import PromptBank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Infer reconstruction with prompts")
    p.add_argument("--predictor_ckpt", type=str, required=True)
    p.add_argument("--recon_ckpt", type=str, required=True)
    p.add_argument("--ir_input", type=str, required=True, help="IR image or directory")
    p.add_argument("--vi_input", type=str, required=True, help="VI image or directory")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--Kir", type=int, required=True)
    p.add_argument("--Kvi", type=int, required=True)
    p.add_argument("--D", type=int, default=256)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_json", type=str, default="")
    return p.parse_args()


def _collect(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _load_gray(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    arr = arr.view(img.size[1], img.size[0]).float() / 255.0
    return arr.unsqueeze(0).unsqueeze(0).to(device)


def _save_gray(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (x.squeeze().clamp(0, 1).cpu().numpy() * 255.0).astype("uint8")
    Image.fromarray(arr, mode="L").save(path)


def _build_predictor(ckpt: str, kir: int, kvi: int, device: torch.device) -> HierarchicalPromptPredictor:
    payload = torch.load(ckpt, map_location=device)
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


def _topk(p: torch.Tensor, k: int) -> list[dict]:
    values, idx = torch.topk(p, k=min(k, p.numel()))
    return [{"idx": int(i.item()), "prob": float(v.item())} for v, i in zip(values, idx)]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    k = args.Kir + args.Kvi

    predictor = _build_predictor(args.predictor_ckpt, args.Kir, args.Kvi, device)

    recon_ckpt = torch.load(args.recon_ckpt, map_location=device)
    recon_args = recon_ckpt.get("args", {})
    prompt_bank = PromptBank(k=k, d=args.D).to(device)
    prompt_bank.load_state_dict(recon_ckpt["prompt_bank"], strict=True)

    ae = DINOv2AutoEncoder(
        prompt_dim=args.D,
        dinov2_name=recon_args.get("dinov2_name", "dinov2_vits14"),
        dinov2_local_ckpt=recon_args.get("dinov2_local_ckpt", ""),
    ).to(device)
    ae.load_state_dict(recon_ckpt["autoencoder"], strict=True)
    ae.eval()
    prompt_bank.eval()

    ir_paths = _collect(Path(args.ir_input))
    vi_paths = _collect(Path(args.vi_input))

    if len(ir_paths) != len(vi_paths):
        raise ValueError("ir_input and vi_input must have the same number of files")

    out_dir = Path(args.out_dir)
    records = []
    with torch.no_grad():
        for ir_path, vi_path in zip(ir_paths, vi_paths):
            x_ir = _load_gray(ir_path, device)
            x_vi = _load_gray(vi_path, device)

            p_ir = predictor.predict_P(x_ir)
            pv_ir = prompt_bank(p_ir)
            y_ir = ae(x_ir, pv_ir)

            p_vi = predictor.predict_P(x_vi)
            pv_vi = prompt_bank(p_vi)
            y_vi = ae(x_vi, pv_vi)

            _save_gray(y_ir, out_dir / "ir" / ir_path.name)
            _save_gray(y_vi, out_dir / "vi" / vi_path.name)

            rec = {
                "ir_name": ir_path.name,
                "vi_name": vi_path.name,
                "P_ir_topk": _topk(p_ir[0].cpu(), args.topk),
                "P_vi_topk": _topk(p_vi[0].cpu(), args.topk),
            }
            records.append(rec)
            print(f"saved {ir_path.name} and {vi_path.name}")

    if args.save_json:
        out_json = Path(args.save_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
