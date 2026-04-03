#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
from uie.models import FastUNetEnhancer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FastUNetEnhancer(base=args.base).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    example = torch.randn(1,3,args.h,args.w, device=device).clamp(0,1)
    ts = torch.jit.trace(model, example)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    ts.save(str(out))
    print("Saved TorchScript:", out)

if __name__ == "__main__":
    main()
