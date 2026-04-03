#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2lab
from skimage import img_as_float32
import lpips

from uie.models import FastUNetEnhancer
from uie.data import PairDataset

# ---------------------------
# Helpers
# ---------------------------
def tensor_to_np(t):
    x = t.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(x, 0.0, 1.0).astype(np.float32)

def to_gray_uint8(rgb01):
    gray = cv2.cvtColor((rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray

# ---------------------------
# MS-SSIM
# ---------------------------
def _ssim_single(x, y, data_range=1.0, win=11):
    return structural_similarity(x, y, channel_axis=-1, data_range=data_range)

def ms_ssim_np(x, y, levels=5):
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    xs, ys = x.copy(), y.copy()
    mssim = []
    for _ in range(levels):
        mssim.append(structural_similarity(xs, ys, channel_axis=-1, data_range=1.0))
        if min(xs.shape[0], xs.shape[1]) < 2 or min(ys.shape[0], ys.shape[1]) < 2:
            break
        xs = cv2.pyrDown(xs)
        ys = cv2.pyrDown(ys)
    weights = weights[:len(mssim)]
    if len(mssim) == 1:
        return float(mssim[0])
    score = 1.0
    for w, s in zip(weights, mssim):
        score *= float(max(s, 1e-6)) ** w
    return float(score)

# ---------------------------
# NIQE
# Simple fallback implementation based on skimage if available;
# otherwise returns NaN.
# ---------------------------
try:
    from skimage.metrics import normalized_root_mse  # dummy import to test skimage
    from skimage import metrics as skm
    _HAS_SKI = True
except Exception:
    _HAS_SKI = False

def niqe_fallback(rgb01):
    # If no NIQE implementation available, return NaN.
    # You can later replace with pyiqa-based NIQE if desired.
    try:
        from skimage.metrics import niqe
        gray = to_gray_uint8(rgb01)
        return float(niqe(gray))
    except Exception:
        return float("nan")

# ---------------------------
# UIQM
# ---------------------------
def _uicm(img):
    R = img[:, :, 0].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    B = img[:, :, 2].astype(np.float32)
    RG = R - G
    YB = 0.5 * (R + G) - B

    def mu_a(x, alpha_L=0.1, alpha_R=0.1):
        x = np.sort(x.flatten())
        K = len(x)
        T_a_L = int(alpha_L * K)
        T_a_R = int(alpha_R * K)
        if K - T_a_L - T_a_R <= 0:
            return float(np.mean(x))
        return float(np.mean(x[T_a_L:K - T_a_R]))

    def s_a(x, mu):
        x = x.flatten()
        return float(np.mean((x - mu) ** 2))

    mu_rg = mu_a(RG)
    mu_yb = mu_a(YB)
    sig_rg = math.sqrt(max(s_a(RG, mu_rg), 1e-12))
    sig_yb = math.sqrt(max(s_a(YB, mu_yb), 1e-12))
    return (-0.0268 * math.sqrt(mu_rg**2 + mu_yb**2)) + (0.1586 * math.sqrt(sig_rg**2 + sig_yb**2))

def _eme(channel, block_size=8):
    H, W = channel.shape
    k1 = H // block_size
    k2 = W // block_size
    if k1 == 0 or k2 == 0:
        return 0.0
    val = 0.0
    eps = 1e-6
    for i in range(k1):
        for j in range(k2):
            block = channel[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            Imax = float(np.max(block))
            Imin = float(np.min(block))
            val += math.log((Imax + eps) / (Imin + eps) + eps)
    return 2.0 * val / (k1 * k2)

def _uism(img):
    # Sobel-based sharpness
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    mag = np.clip(mag, 0, 1)
    return _eme(mag, block_size=8)

def _uiconm(img, block_size=8):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = gray.shape
    k1 = H // block_size
    k2 = W // block_size
    if k1 == 0 or k2 == 0:
        return 0.0
    val = 0.0
    eps = 1e-6
    for i in range(k1):
        for j in range(k2):
            block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            Imax = float(np.max(block))
            Imin = float(np.min(block))
            val += (Imax - Imin) / (Imax + Imin + eps)
    return val / (k1 * k2)

def uiqm(img):
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return c1 * _uicm(img) + c2 * _uism(img) + c3 * _uiconm(img)

# ---------------------------
# UCIQE
# ---------------------------
def uciqe(img):
    # img in [0,1], RGB
    lab = rgb2lab(img_as_float32(img))
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    chroma = np.sqrt(a**2 + b**2)

    sigma_c = np.std(chroma)
    con_l = np.percentile(L, 99) - np.percentile(L, 1)

    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    mu_s = np.mean(sat)

    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--base", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = PairDataset(args.pairs_csv, crop=384, training=False)
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)

    model = FastUNetEnhancer(base=args.base).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)

    # Prefer EMA if present
    if isinstance(ckpt, dict) and ckpt.get("ema") is not None:
        model.load_state_dict(ckpt["ema"], strict=True)
    elif isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    model.eval()

    lp = lpips.LPIPS(net="alex").to(device)
    lp.eval()

    rows = []
    psnrs, ssims, msssims, lpips_vals, niqes, uiqms, uciqes = [], [], [], [], [], [], []

    with torch.no_grad():
        for x, y, name in ld:
            x = x.to(device)
            y = y.to(device)

            pred = model(x).clamp(0, 1)

            pred_np = tensor_to_np(pred)
            gt_np = tensor_to_np(y)

            p = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
            s = structural_similarity(gt_np, pred_np, channel_axis=-1, data_range=1.0)
            ms = ms_ssim_np(gt_np, pred_np)

            pred_lp = pred * 2 - 1
            gt_lp = y * 2 - 1
            l = float(lp(pred_lp, gt_lp).mean().item())

            n = niqe_fallback(pred_np)
            uqm = uiqm(pred_np)
            ucq = uciqe(pred_np)

            psnrs.append(p)
            ssims.append(s)
            msssims.append(ms)
            lpips_vals.append(l)
            niqes.append(n)
            uiqms.append(uqm)
            uciqes.append(ucq)

            rows.append([name[0], p, s, ms, l, n, uqm, ucq])

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "PSNR", "SSIM", "MS-SSIM", "LPIPS", "NIQE", "UIQM", "UCIQE"])
        w.writerows(rows)

    def mean_clean(v):
        v = [x for x in v if not (isinstance(x, float) and math.isnan(x))]
        return float(np.mean(v)) if len(v) else float("nan")

    print("Saved:", out)
    print(f"PSNR    ↑ : {mean_clean(psnrs):.4f}")
    print(f"SSIM    ↑ : {mean_clean(ssims):.4f}")
    print(f"MS-SSIM ↑ : {mean_clean(msssims):.4f}")
    print(f"LPIPS   ↓ : {mean_clean(lpips_vals):.4f}")
    print(f"NIQE    ↓ : {mean_clean(niqes):.4f}")
    print(f"UIQM    ↑ : {mean_clean(uiqms):.4f}")
    print(f"UCIQE   ↑ : {mean_clean(uciqes):.4f}")

if __name__ == "__main__":
    main()
