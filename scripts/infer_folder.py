# Copyright 2026 Shahid Ahamed Hasib
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

from uie.models import FastUNetEnhancer


def load_rgb(p: Path):
    return Image.open(p).convert("RGB")


def to_tensor(img: Image.Image):
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t


def to_pil(t: torch.Tensor):
    t = t.detach().clamp(0, 1).cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=32)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available -> using CPU")
        device = "cpu"

    # Load checkpoint
    ck = torch.load(args.ckpt, map_location="cpu")
    model = FastUNetEnhancer(base=args.base).to(device).eval()

    if isinstance(ck, dict) and "model" in ck:
        sd = ck["ema"] if (args.use_ema and ck.get("ema") is not None) else ck["model"]
        model.load_state_dict(sd, strict=True)
    else:
        model.load_state_dict(ck, strict=True)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])

    print("Found", len(files), "images")

    for p in tqdm(files, desc="Infer"):
        x = to_tensor(load_rgb(p)).unsqueeze(0).to(device)

        if args.tile and args.tile > 0:
            _, _, H, W = x.shape
            tile = args.tile
            overlap = args.overlap
            stride = tile - overlap
            if stride <= 0:
                raise ValueError("tile must be larger than overlap")

            out = torch.zeros((1, 3, H, W), device=device, dtype=x.dtype)
            wgt = torch.zeros((1, 1, H, W), device=device, dtype=x.dtype)

            for y0 in range(0, H, stride):
                for x0 in range(0, W, stride):
                    y1 = min(y0 + tile, H)
                    x1 = min(x0 + tile, W)
                    ys = max(0, y1 - tile)
                    xs = max(0, x1 - tile)

                    patch = x[..., ys:y1, xs:x1]
                    pred = model(patch).clamp(0, 1)

                    out[..., ys:y1, xs:x1] += pred
                    wgt[..., ys:y1, xs:x1] += 1.0

            pred = (out / wgt.clamp_min(1e-6)).clamp(0, 1)
        else:
            pred = model(x).clamp(0, 1)

        img = to_pil(pred[0])
        img.save(out_dir / p.name)

    print("Saved enhanced images to:", out_dir)


if __name__ == "__main__":
    main()
