#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from uie.models import FastUNetEnhancer


# ----------------------------
# Utils
# ----------------------------
def parse_hw(s: str):
    s = (s or "").strip().lower()
    if not s:
        return None
    if "x" not in s:
        raise ValueError("Expected WxH, e.g. 1920x1080")
    w, h = s.split("x", 1)
    return int(w), int(h)

def load_ckpt(model, ckpt_path: str, use_ema: bool):
    ck = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ck, dict) and "model" in ck:
        sd = ck["ema"] if (use_ema and ck.get("ema") is not None) else ck["model"]
    else:
        sd = ck
    model.load_state_dict(sd, strict=True)

def bgr_to_tensor01(bgr: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).to(torch.float32) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    return x

def tensor01_to_bgr(x: torch.Tensor) -> np.ndarray:
    x = x.clamp(0, 1)[0].permute(1, 2, 0).detach().cpu().numpy()
    rgb = (x * 255.0).round().astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def pad_to_multiple(x: torch.Tensor, m: int = 8):
    _, _, H, W = x.shape
    Hp = ((H + m - 1) // m) * m
    Wp = ((W + m - 1) // m) * m
    pad_h = Hp - H
    pad_w = Wp - W
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


# ----------------------------
# Post-processing (optional)
# ----------------------------
def gray_world_white_balance_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    img = rgb_u8.astype(np.float32)
    mean = img.reshape(-1, 3).mean(axis=0) + 1e-6
    gray = mean.mean()
    gain = gray / mean
    out = img * gain[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)

def clahe_on_luminance_rgb(rgb_u8: np.ndarray, clip=1.6, tilegrid=8) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tilegrid), int(tilegrid)))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)

def edge_mask_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    mag = np.abs(lap)
    p95 = np.percentile(mag, 95) + 1e-6
    m = np.clip(mag / p95, 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 1.0)
    return m

def unsharp_edge_masked_rgb(rgb_u8: np.ndarray, amount=0.35, radius=1.2, edge_bias=0.75) -> np.ndarray:
    img = rgb_u8.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), float(radius))
    sharp = img + float(amount) * (img - blurred)

    m = edge_mask_rgb(rgb_u8)
    m2 = np.clip((m - edge_bias) / max(1e-6, (1.0 - edge_bias)), 0, 1)
    m3 = m2[..., None]
    out = img * (1.0 - m3) + sharp * m3
    return np.clip(out, 0, 255).astype(np.uint8)

def postprocess_bgr(bgr: np.ndarray, do_wb: bool, do_clahe: bool, clahe_clip: float, clahe_grid: int,
                    do_sharp: bool, sharp_amount: float, sharp_radius: float, sharp_edge_bias: float) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if do_wb:
        rgb = gray_world_white_balance_rgb(rgb)
    if do_clahe:
        rgb = clahe_on_luminance_rgb(rgb, clip=clahe_clip, tilegrid=clahe_grid)
    if do_sharp:
        rgb = unsharp_edge_masked_rgb(rgb, amount=sharp_amount, radius=sharp_radius, edge_bias=sharp_edge_bias)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ----------------------------
# Tiled inference (anti-seam)
# ----------------------------
@torch.no_grad()
def enhance_tiled(model, x: torch.Tensor, tile: int, overlap: int, multiple: int = 8,
                  use_amp: bool = True, offset_y: int = 0, offset_x: int = 0) -> torch.Tensor:
    assert x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 3
    _, _, H, W = x.shape

    xpad, _ = pad_to_multiple(x, m=multiple)
    _, _, Hm, Wm = xpad.shape

    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be < tile")

    # pad so tiling covers full
    if Hm <= tile:
        pad_h = tile - Hm
    else:
        pad_h = (math.ceil((Hm - tile) / stride) * stride + tile - Hm)
    if Wm <= tile:
        pad_w = tile - Wm
    else:
        pad_w = (math.ceil((Wm - tile) / stride) * stride + tile - Wm)

    xpad2 = F.pad(xpad, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, Hp, Wp = xpad2.shape

    out  = torch.zeros((1, 3, Hp, Wp), device=x.device, dtype=torch.float32)
    wsum = torch.zeros((1, 1, Hp, Wp), device=x.device, dtype=torch.float32)

    # smooth blend window
    yy = torch.linspace(0, math.pi, tile, device=x.device).view(1, 1, tile, 1)
    xx = torch.linspace(0, math.pi, tile, device=x.device).view(1, 1, 1, tile)
    win = (torch.sin(yy) * torch.sin(xx)).clamp_min(1e-6)

    oy = (offset_y % stride)
    ox = (offset_x % stride)

    ys = list(range(oy, Hp - tile + 1, stride))
    xs = list(range(ox, Wp - tile + 1, stride))
    if 0 not in ys: ys = [0] + ys
    if 0 not in xs: xs = [0] + xs
    ys = sorted(set(ys))
    xs = sorted(set(xs))

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and x.is_cuda))

    for top in ys:
        for left in xs:
            patch = xpad2[:, :, top:top+tile, left:left+tile]
            if patch.shape[-2] != tile or patch.shape[-1] != tile:
                continue
            with amp_ctx:
                pred = model(patch).float().clamp(0, 1)
            out[:, :, top:top+tile, left:left+tile] += pred * win
            wsum[:, :, top:top+tile, left:left+tile] += win

    out = out / wsum.clamp_min(1e-6)
    out = out[:, :, :Hm, :Wm]
    out = out[:, :, :H, :W]
    return out

@torch.no_grad()
def enhance_two_pass(model, x: torch.Tensor, tile: int, overlap: int, multiple: int, use_amp: bool) -> torch.Tensor:
    stride = tile - overlap
    oy = stride // 2
    ox = stride // 2
    y1 = enhance_tiled(model, x, tile=tile, overlap=overlap, multiple=multiple, use_amp=use_amp, offset_y=0,  offset_x=0)
    y2 = enhance_tiled(model, x, tile=tile, overlap=overlap, multiple=multiple, use_amp=use_amp, offset_y=oy, offset_x=ox)
    return (0.5 * y1 + 0.5 * y2).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_video", required=True)
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_fps", type=float, default=0.0, help="0 = keep input fps")
    ap.add_argument("--resize", default="", help="e.g. 1920x1080 or leave empty")

    # inference controls
    ap.add_argument("--tiled", action="store_true", help="use tiled inference")
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--two_pass", action="store_true", help="shifted second pass to reduce seams/artifacts")
    ap.add_argument("--multiple", type=int, default=8)
    ap.add_argument("--amp", action="store_true", help="use CUDA AMP fp16")

    # post
    ap.add_argument("--wb", action="store_true")
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--clahe_clip", type=float, default=1.6)
    ap.add_argument("--clahe_grid", type=int, default=8)
    ap.add_argument("--sharpen", action="store_true")
    ap.add_argument("--sharp_amount", type=float, default=0.35)
    ap.add_argument("--sharp_radius", type=float, default=1.2)
    ap.add_argument("--sharp_edge_bias", type=float, default=0.75)

    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    model = FastUNetEnhancer(base=args.base).to(device).eval()
    load_ckpt(model, args.ckpt, use_ema=args.use_ema)

    cap = cv2.VideoCapture(args.in_video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.in_video}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    in_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_size = parse_hw(args.resize)
    out_w, out_h = (out_size if out_size else (in_w, in_h))

    out_fps = in_fps
    if args.max_fps and args.max_fps > 0:
        out_fps = min(out_fps, float(args.max_fps))

    out_path = Path(args.out_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_w, out_h))
    if not vw.isOpened():
        raise SystemExit(f"Could not open output: {out_path}")

    print("Input:", args.in_video)
    print(f"Input size: {in_w}x{in_h} @ {in_fps:.2f} fps")
    print(f"Output: {out_w}x{out_h} @ {out_fps:.2f} fps")
    print(f"Tiled={args.tiled} tile={args.tile} overlap={args.overlap} two_pass={args.two_pass} amp={args.amp}")
    print(f"Post: wb={args.wb} clahe={args.clahe} sharpen={args.sharpen}")

    frame_i = 0
    t0 = time.time()
    keep_every = 1
    if args.max_fps and args.max_fps > 0 and in_fps > 0:
        keep_every = max(1, int(round(in_fps / out_fps)))

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame_i += 1

        # frame skip for max_fps
        if keep_every > 1 and (frame_i % keep_every) != 0:
            continue

        if (in_w, in_h) != (out_w, out_h):
            bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

        x = bgr_to_tensor01(bgr, device=device)

        with torch.no_grad():
            if args.tiled:
                if args.two_pass:
                    y = enhance_two_pass(model, x, tile=args.tile, overlap=args.overlap,
                                        multiple=args.multiple, use_amp=args.amp)
                else:
                    y = enhance_tiled(model, x, tile=args.tile, overlap=args.overlap,
                                      multiple=args.multiple, use_amp=args.amp)
            else:
                amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.amp and device=="cuda"))
                with amp_ctx:
                    y = model(x).float().clamp(0, 1)

        out_bgr = tensor01_to_bgr(y)

        if args.wb or args.clahe or args.sharpen:
            out_bgr = postprocess_bgr(
                out_bgr,
                do_wb=args.wb,
                do_clahe=args.clahe,
                clahe_clip=args.clahe_clip,
                clahe_grid=args.clahe_grid,
                do_sharp=args.sharpen,
                sharp_amount=args.sharp_amount,
                sharp_radius=args.sharp_radius,
                sharp_edge_bias=args.sharp_edge_bias,
            )

        vw.write(out_bgr)

        if frame_i % 60 == 0:
            dt = time.time() - t0
            print(f"processed {frame_i} frames in {dt:.1f}s ({frame_i/max(dt,1e-6):.2f} fps input loop)")

    cap.release()
    vw.release()
    dt = time.time() - t0
    print(f"Done. Wrote: {out_path}")
    print(f"Total time: {dt:.1f}s, frames read: {frame_i}")

if __name__ == "__main__":
    main()
