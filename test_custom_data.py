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

!pip -q install torch torchvision opencv-python pillow tqdm

import os, glob, math
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConv(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, k, stride=s, padding=p, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        mid = max(1, c // r)
        self.fc1 = nn.Conv2d(c, mid, 1)
        self.fc2 = nn.Conv2d(mid, c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class Block(nn.Module):
    def __init__(self, c_in, c_out, down=False):
        super().__init__()
        stride = 2 if down else 1
        self.c1 = DWConv(c_in, c_out, s=stride)
        self.c2 = DWConv(c_out, c_out, s=1)
        self.se = SE(c_out)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return self.se(x)

class UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.fuse = Block(c_in + c_skip, c_out, down=False)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)

class FastUNetEnhancer(nn.Module):
    def __init__(self, base=32, residual_alpha=0.05):
        super().__init__()
        b = base
        self.residual_alpha = float(residual_alpha)

        self.e0 = Block(3, b, down=False)
        self.e1 = Block(b, b*2, down=True)
        self.e2 = Block(b*2, b*4, down=True)
        self.e3 = Block(b*4, b*6, down=True)
        self.mid = Block(b*6, b*6, down=False)

        self.u2 = UpBlock(b*6, b*6, b*4)
        self.u1 = UpBlock(b*4, b*4, b*2)
        self.u0 = UpBlock(b*2, b*2, b)

        self.head = nn.Conv2d(b, 3, 1)

    def forward(self, x):
        x_in = x

        s0 = self.e0(x)
        s1 = self.e1(s0)
        s2 = self.e2(s1)
        s3 = self.e3(s2)

        m  = self.mid(s3)
        d2 = self.u2(m, s3)
        d1 = self.u1(d2, s2)
        d0 = self.u0(d1, s1)

        d0 = F.interpolate(d0, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        out = torch.sigmoid(self.head(d0))

        # keep same behavior you were using
        a = self.residual_alpha
        out = (1.0 - a) * out + a * x_in
        return out.clamp(0.0, 1.0)


ckpt_path = "/content/best.pt"
inp_dir   = "/content/images"
out_dir   = "/content/enhanced_outputs_raw_model_only"

# robust tiling settings
TILE_BASE    = 768
OVERLAP_BASE = 256
MULTIPLE     = 8
USE_AMP      = True
USE_TWO_PASS = True

# gallery settings
THUMB_W = 320
THUMB_H = 240
GAP     = 12
LABEL_H = 32

os.makedirs(out_dir, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

ckpt = torch.load(ckpt_path, map_location="cpu")
if isinstance(ckpt, dict) and ckpt.get("ema") is not None:
    state = ckpt["ema"]
    print("Loaded: ckpt['ema']")
elif isinstance(ckpt, dict) and "model" in ckpt:
    state = ckpt["model"]
    print("Loaded: ckpt['model']")
else:
    state = ckpt
    print("Loaded: raw/state dict")

model = FastUNetEnhancer(base=32, residual_alpha=0.05).to(device)
model.load_state_dict(state, strict=True)
model.eval()

if device == "cuda":
    torch.backends.cudnn.benchmark = True


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def tensor_to_np_uint8(t: torch.Tensor) -> np.ndarray:
    t = t.clamp(0, 1)[0].permute(1, 2, 0).detach().cpu().numpy()
    return (t * 255.0 + 0.5).astype(np.uint8)

def safe_pad(x: torch.Tensor, pad_l, pad_r, pad_t, pad_b, mode="reflect") -> torch.Tensor:
    _, _, H, W = x.shape
    use_mode = mode
    if mode == "reflect":
        if (pad_t >= H) or (pad_b >= H) or (pad_l >= W) or (pad_r >= W):
            use_mode = "replicate"
    return F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=use_mode)

def pad_to_multiple(x: torch.Tensor, m: int = 8):
    _, _, H, W = x.shape
    Hp = ((H + m - 1) // m) * m
    Wp = ((W + m - 1) // m) * m
    pad_h = Hp - H
    pad_w = Wp - W
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = safe_pad(x, 0, pad_w, 0, pad_h, mode="reflect")
    return x, (pad_h, pad_w)

def fit_with_pad(img_u8, out_w, out_h, bg=(255, 255, 255)):
    im = Image.fromarray(img_u8)
    im.thumbnail((out_w, out_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (out_w, out_h), bg)
    x = (out_w - im.width) // 2
    y = (out_h - im.height) // 2
    canvas.paste(im, (x, y))
    return np.array(canvas)

def add_text_bar(img_u8, text, bar_h=32, bg=(20, 20, 20), fg=(255, 255, 255)):
    H, W, _ = img_u8.shape
    bar = Image.new("RGB", (W, bar_h), bg)
    draw = ImageDraw.Draw(bar)
    draw.text((10, 8), text, fill=fg)
    combined = Image.new("RGB", (W, H + bar_h))
    combined.paste(bar, (0, 0))
    combined.paste(Image.fromarray(img_u8), (0, bar_h))
    return np.array(combined)

def make_single_2row_gallery(raw_items, model_items, names, thumb_w=320, thumb_h=240, gap=12, label_h=32):
    assert len(raw_items) == len(model_items) == len(names)
    n = len(names)
    if n == 0:
        return None

    raw_cells = []
    model_cells = []

    for raw_u8, model_u8, name in zip(raw_items, model_items, names):
        raw_fit = fit_with_pad(raw_u8, thumb_w, thumb_h)
        mdl_fit = fit_with_pad(model_u8, thumb_w, thumb_h)

        raw_cell = add_text_bar(raw_fit, f"RAW - {name}", bar_h=label_h)
        mdl_cell = add_text_bar(mdl_fit, f"MODEL OUTPUT - {name}", bar_h=label_h)

        raw_cells.append(raw_cell)
        model_cells.append(mdl_cell)

    cell_h = thumb_h + label_h
    total_w = n * thumb_w + (n - 1) * gap
    total_h = 2 * cell_h + gap

    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

    # top row = raw
    x = 0
    for cell in raw_cells:
        canvas[0:cell_h, x:x+thumb_w] = cell
        x += thumb_w + gap

    # bottom row = model output
    x = 0
    y0 = cell_h + gap
    for cell in model_cells:
        canvas[y0:y0+cell_h, x:x+thumb_w] = cell
        x += thumb_w + gap

    return canvas


@torch.no_grad()
def enhance_tiled(x: torch.Tensor, tile, overlap, multiple=8, use_amp=True, offset_y=0, offset_x=0):
    assert x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 3
    _, _, H, W = x.shape

    xpad, _ = pad_to_multiple(x, m=multiple)
    _, _, Hm, Wm = xpad.shape

    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be < tile")

    if Hm <= tile:
        pad_h = tile - Hm
    else:
        pad_h = (math.ceil((Hm - tile) / stride) * stride + tile - Hm)

    if Wm <= tile:
        pad_w = tile - Wm
    else:
        pad_w = (math.ceil((Wm - tile) / stride) * stride + tile - Wm)

    xpad2 = safe_pad(xpad, 0, pad_w, 0, pad_h, mode="reflect")
    _, _, Hp, Wp = xpad2.shape

    out  = torch.zeros((1, 3, Hp, Wp), device=x.device, dtype=torch.float32)
    wsum = torch.zeros((1, 1, Hp, Wp), device=x.device, dtype=torch.float32)

    yy = torch.linspace(0, math.pi, tile, device=x.device).view(1, 1, tile, 1)
    xx = torch.linspace(0, math.pi, tile, device=x.device).view(1, 1, 1, tile)
    win = (torch.sin(yy) * torch.sin(xx)).clamp_min(1e-6)

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and x.is_cuda))

    start_y = int(offset_y) % stride
    start_x = int(offset_x) % stride

    ys = list(range(start_y, Hp - tile + 1, stride))
    xs = list(range(start_x, Wp - tile + 1, stride))
    if 0 not in ys:
        ys = [0] + ys
    if 0 not in xs:
        xs = [0] + xs
    ys = sorted(set(ys))
    xs = sorted(set(xs))

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
def enhance_two_pass(x: torch.Tensor, tile, overlap):
    stride = tile - overlap
    oy = stride // 2
    ox = stride // 2
    y1 = enhance_tiled(x, tile=tile, overlap=overlap, multiple=MULTIPLE, use_amp=USE_AMP, offset_y=0,  offset_x=0)
    y2 = enhance_tiled(x, tile=tile, overlap=overlap, multiple=MULTIPLE, use_amp=USE_AMP, offset_y=oy, offset_x=ox)
    return (0.5 * y1 + 0.5 * y2).clamp(0, 1)

@torch.no_grad()
def run_model_robust(x: torch.Tensor):
    _, _, H, W = x.shape

    tile = min(TILE_BASE, max(128, (min(H, W) // 32) * 32))
    tile = int(tile)

    overlap = min(OVERLAP_BASE, tile // 3)
    overlap = int(overlap)

    if H <= tile and W <= tile:
        xpad, _ = pad_to_multiple(x, m=MULTIPLE)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(USE_AMP and x.is_cuda)):
            y = model(xpad).float().clamp(0, 1)
        return y[:, :, :H, :W]

    if USE_TWO_PASS:
        return enhance_two_pass(x, tile=tile, overlap=overlap)
    else:
        return enhance_tiled(x, tile=tile, overlap=overlap, multiple=MULTIPLE, use_amp=USE_AMP)


paths = sorted(glob.glob(os.path.join(inp_dir, "*.*")))
print("Found images:", len(paths))
print("Saving to:", out_dir)

all_raw = []
all_model = []
all_names = []

for p in tqdm(paths):
    img_pil = Image.open(p).convert("RGB")
    inp_u8  = np.asarray(img_pil).astype(np.uint8)

    x = pil_to_tensor(img_pil).to(device)

    # model inference only
    y = run_model_robust(x)
    model_u8 = tensor_to_np_uint8(y)

    base = os.path.splitext(os.path.basename(p))[0]

    out_raw   = os.path.join(out_dir, f"{base}_raw.png")
    out_model = os.path.join(out_dir, f"{base}_model.png")

    Image.fromarray(inp_u8).save(out_raw)
    Image.fromarray(model_u8).save(out_model)

    all_raw.append(inp_u8)
    all_model.append(model_u8)
    all_names.append(base)

# save one single 2-row image with all results
gallery = make_single_2row_gallery(
    all_raw,
    all_model,
    all_names,
    thumb_w=THUMB_W,
    thumb_h=THUMB_H,
    gap=GAP,
    label_h=LABEL_H
)

if gallery is not None:
    gallery_path = os.path.join(out_dir, "all_results_2rows.png")
    Image.fromarray(gallery).save(gallery_path)
    print("Saved gallery:", gallery_path)

print("Done. Check:", out_dir)

# zip results
zip_path = "/content/enhanced_outputs_raw_model_only.zip"
!zip -r {zip_path} {out_dir}

from google.colab import files
files.download(zip_path)
