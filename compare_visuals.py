#!/usr/bin/env python3
import argparse, random
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

from uie.models import FastUNetEnhancer

IMG_EXTS = {".png",".jpg",".jpeg"}

def read_rgb(p):
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None: return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def write_rgb(p, rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(p), bgr)

def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 1, cv2.LINE_AA)
    return out

@torch.no_grad()
def infer(model, rgb, device, max_side=960):
    h,w = rgb.shape[:2]
    scale = 1.0
    if max(h,w) > max_side:
        scale = max_side / max(h,w)
        rgb_s = cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        rgb_s = rgb

    x = torch.from_numpy(rgb_s.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
    y = model(x).clamp(0,1)
    out = (y.squeeze(0).permute(1,2,0).cpu().numpy()*255.0).astype(np.uint8)

    if scale != 1.0:
        out = cv2.resize(out, (w,h), interpolation=cv2.INTER_LINEAR)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--ours_ckpt", required=True)
    ap.add_argument("--lite_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--base", type=int, default=32)
    args = ap.parse_args()

    inp = Path(args.inp_dir)
    gt  = Path(args.gt_dir)
    lite= Path(args.lite_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    files = [p for p in inp.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files = sorted(files)
    random.shuffle(files)
    files = files[:args.n]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FastUNetEnhancer(base=args.base).to(device)
    ckpt = torch.load(args.ours_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    for p in tqdm(files, desc="Visual compare"):
        name = p.name
        a = read_rgb(p)
        b = read_rgb(gt/name)
        c = read_rgb(lite/name)
        if a is None or b is None or c is None:
            continue

        ours = infer(model, a, device)

        g1 = np.concatenate([label(a,"Input"), label(ours,"Ours")], axis=1)
        g2 = np.concatenate([label(c,"LiteEnhanceNet"), label(b,"GT")], axis=1)
        grid = np.concatenate([g1,g2], axis=0)
        write_rgb(out/f"cmp_{name}", grid)

    print("Saved grids in:", out)

if __name__ == "__main__":
    main()
