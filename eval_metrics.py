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
import argparse, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

from uie.models import FastUNetEnhancer
from uie.data import PairDataset

def t2np(t):
    x = t.squeeze(0).detach().cpu().numpy().transpose(1,2,0)
    return np.clip(x, 0, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--base", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PairDataset(args.pairs_csv, crop=256, training=False)
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)

    model = FastUNetEnhancer(base=args.base).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    lp = lpips.LPIPS(net="alex").to(device)
    lp.eval()

    rows = []
    psnrs, ssims, lps = [], [], []

    with torch.no_grad():
        for x, y, name in ld:
            x = x.to(device); y = y.to(device)
            pred = model(x).clamp(0,1)

            pred_np = t2np(pred)
            gt_np   = t2np(y)

            p = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
            s = structural_similarity(gt_np, pred_np, channel_axis=-1, data_range=1.0)

            # LPIPS expects [-1,1]
            pred_lp = (pred*2 - 1)
            gt_lp   = (y*2 - 1)
            l = float(lp(pred_lp, gt_lp).mean().item())

            psnrs.append(p); ssims.append(s); lps.append(l)
            rows.append([name[0], p, s, l])

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name","psnr","ssim","lpips"])
        w.writerows(rows)

    print("Saved:", out)
    print(f"Mean PSNR={float(np.mean(psnrs)):.3f}  Mean SSIM={float(np.mean(ssims)):.4f}  Mean LPIPS={float(np.mean(lps)):.4f}")

if __name__ == "__main__":
    main()
