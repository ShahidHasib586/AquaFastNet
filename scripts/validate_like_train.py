#!/usr/bin/env python3
import argparse
from pathlib import Path
import hashlib

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from uie.models import FastUNetEnhancer
from uie.data import PairDataset

def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2).clamp_min(1e-12)
    return (10.0 * torch.log10(1.0 / mse)).item()

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--crop", type=int, default=384, help="not used when training=False, but keep for parity")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--num_workers", type=int, default=1)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    ckpt_path = Path(args.ckpt)

    print("CSV :", csv_path)
    print("CKPT:", ckpt_path)
    if csv_path.exists():
        print("CSV lines:", sum(1 for _ in csv_path.open("r", encoding="utf-8", errors="ignore")))
        print("CSV sha256:", sha256_file(csv_path))
    else:
        raise SystemExit(f"Missing CSV: {csv_path}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available -> using CPU")
        device = "cpu"

    # dataset exactly like train.py validation
    val_ds = PairDataset(str(csv_path), crop=args.crop, training=False)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # model
    model = FastUNetEnhancer(base=args.base).to(device).eval()

    # load ckpt exactly like your train.py structure
    ck = torch.load(str(ckpt_path), map_location="cpu")
    if not (isinstance(ck, dict) and "model" in ck):
        raise SystemExit("This ckpt does not look like your train.py checkpoint (missing 'model').")

    sd = ck["ema"] if (args.use_ema and ck.get("ema") is not None) else ck["model"]
    model.load_state_dict(sd, strict=True)

    vals = []
    for x, y, name in tqdm(val_ld, desc="Val"):
        x = x.to(device)
        y = y.to(device)
        pred = model(x).clamp(0, 1)
        vals.append(psnr(pred, y))

    mean_psnr = sum(vals) / max(1, len(vals))
    print("\nRESULT")
    print("epoch:", ck.get("epoch"))
    print("stored best:", ck.get("best"))
    print("use_ema:", bool(args.use_ema))
    print("MEAN_PSNR(train.py-style):", f"{mean_psnr:.6f}")

if __name__ == "__main__":
    main()
