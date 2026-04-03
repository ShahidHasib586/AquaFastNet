#!/usr/bin/env python3
import argparse, csv, math, os, sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch


from uie.models import FastUNetEnhancer


def load_rgb(path: str):
    return Image.open(path).convert("RGB")


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)


def psnr_torch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    # matches train.py logic
    mse = torch.mean((pred - gt) ** 2).clamp_min(1e-12)
    return (10.0 * torch.log10(1.0 / mse)).item()


@torch.no_grad()
def run_full(model, x, tile: int = 0, overlap: int = 32):
    """
    x: (1,3,H,W) in [0,1]
    If tile>0, run tiled inference to avoid OOM.
    """
    if tile <= 0:
        return model(x).clamp(0, 1)

    _, _, h, w = x.shape
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("tile must be > overlap")

    out = torch.zeros((1, 3, h, w), device=x.device, dtype=x.dtype)
    wgt = torch.zeros((1, 1, h, w), device=x.device, dtype=x.dtype)

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            ys = max(0, y1 - tile)
            xs = max(0, x1 - tile)

            patch = x[..., ys:y1, xs:x1]
            pred = model(patch).clamp(0, 1)

            out[..., ys:y1, xs:x1] += pred
            wgt[..., ys:y1, xs:x1] += 1.0

    return (out / wgt.clamp_min(1e-6)).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: input,gt")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tile", type=int, default=0, help="0=full image, else tile size")
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--use_ema", action="store_true", help="Use EMA weights if present")
    ap.add_argument("--limit", type=int, default=0, help="0=all, else evaluate first N rows")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load ckpt
    ck = torch.load(args.ckpt, map_location="cpu")

    model = FastUNetEnhancer(base=args.base).to(device).eval()

    if isinstance(ck, dict) and "model" in ck:
        sd = ck["ema"] if (args.use_ema and ck.get("ema") is not None) else ck["model"]
        model.load_state_dict(sd, strict=True)
    else:
        model.load_state_dict(ck, strict=True)

    # Read CSV
    rows = []
    with open(args.csv, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row["input"], row["gt"]))

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    psnrs = []
    for i, (inp, gt) in enumerate(rows, start=1):
        x = to_tensor(load_rgb(inp)).unsqueeze(0).to(device)
        y = to_tensor(load_rgb(gt)).unsqueeze(0).to(device)

        pred = run_full(model, x, tile=args.tile, overlap=args.overlap)
        ps = psnr_torch(pred, y)
        psnrs.append(ps)

        if i % 50 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] PSNR={sum(psnrs)/len(psnrs):.4f}")

    mean_psnr = sum(psnrs) / max(1, len(psnrs))
    print("CKPT:", args.ckpt)
    print("CSV :", args.csv)
    print("N   :", len(psnrs))
    print(f"MEAN_PSNR: {mean_psnr:.6f}")


if __name__ == "__main__":
    main()