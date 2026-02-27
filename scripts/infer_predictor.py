"""Inference script for hierarchical prompt predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

try:
    from torchvision import transforms
except ImportError as exc:
    raise ImportError("torchvision is required for infer_predictor.py") from exc

from predictors.hierarchical_predictor import HierarchicalPromptPredictor, PredictorConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer hierarchical predictor")
    p.add_argument("--input", type=str, required=True, help="Image path or folder")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--label_maps", type=str, default="")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--output_json", type=str, default="")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_label_maps(path: str | None, ckpt: dict) -> dict:
    if path and Path(path).is_file():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    if "label_maps" in ckpt:
        return ckpt["label_maps"]
    raise ValueError("label maps not found; pass --label_maps")


def collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def topk_dict(prob: torch.Tensor, names: list[str], k: int) -> list[dict]:
    vals, idx = torch.topk(prob, k=min(k, prob.numel()))
    return [{"name": names[i.item()], "prob": float(v.item())} for v, i in zip(vals, idx)]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    label_maps = load_label_maps(args.label_maps if args.label_maps else None, ckpt)

    ir_names = [k for k, _ in sorted(label_maps["ir"].items(), key=lambda kv: kv[1])]
    vi_names = [k for k, _ in sorted(label_maps["vi"].items(), key=lambda kv: kv[1])]
    kir, kvi = label_maps["Kir"], label_maps["Kvi"]

    saved_args = ckpt.get("args", {})
    cfg = PredictorConfig(
        feat_dim=saved_args.get("feat_dim", 512),
        kir=kir,
        kvi=kvi,
        use_clip=saved_args.get("use_clip", False),
        clip_model_name=saved_args.get("clip_model_name", "ViT-H-14"),
        freeze_clip=saved_args.get("freeze_clip", True),
        temperature=saved_args.get("temperature", 1.0),
        label_smoothing=saved_args.get("label_smoothing", 0.0),
    )
    model = HierarchicalPromptPredictor(cfg).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    input_paths = collect_inputs(Path(args.input))
    if not input_paths:
        raise ValueError("No input images found")

    results = []
    with torch.no_grad():
        for path in input_paths:
            image = tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            out = model(images=image)
            p_mod = out["p_mod"][0].cpu()
            p_ir = out["p_deg_ir"][0].cpu()
            p_vi = out["p_deg_vi"][0].cpu()
            p_global = out["P"][0].cpu()

            mod_pred = int(torch.argmax(p_mod).item())
            mod_name = "ir" if mod_pred == 0 else "vi"
            deg_topk = topk_dict(p_ir if mod_pred == 0 else p_vi, ir_names if mod_pred == 0 else vi_names, args.topk)

            item = {
                "path": str(path),
                "p_mod": {"ir": float(p_mod[0].item()), "vi": float(p_mod[1].item())},
                "pred_modality": mod_name,
                "p_deg_ir": {name: float(p_ir[i].item()) for i, name in enumerate(ir_names)},
                "p_deg_vi": {name: float(p_vi[i].item()) for i, name in enumerate(vi_names)},
                "P": p_global.tolist(),
                "topk_deg_for_pred_mod": deg_topk,
            }
            print(f"[{path.name}] p_mod={item['p_mod']} pred_modality={mod_name} topk={deg_topk}")
            print(f"[{path.name}] P={item['P']}")
            results.append(item)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved json to {out_path}")


if __name__ == "__main__":
    main()
