# ============================
# FULL COPY-PASTE TEST SCRIPT (ROBUST + SIDE-BY-SIDE)
# Saves 3 files per input:
#   1) *_raw_model.png     (model output only)
#   2) *_post.png          (model output + safe postprocess)
#   3) *_compare.png       (side-by-side: input | model | post)
#
# Robust for ANY image size:
# - auto-adjusts tile if image is smaller than TILE
# - safe padding/cropping (no reflect-pad errors)
# - no "padding error" even for tiny images
# ============================

!pip -q install torch torchvision opencv-python pillow tqdm

import os, glob, math
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1) Model definition (FIXED residual blend; no burning)
# ============================================================
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

        # ✅ FIX: convex blend (no additive burn)
        a = self.residual_alpha
        out = (1.0 - a) * out + a * x_in
        return out.clamp(0.0, 1.0)

# ============================================================
# 2) USER SETTINGS (EDIT THESE)
# ============================================================
ckpt_path = "/content/best184.pt"
inp_dir   = "/content/images1"
out_dir   = "/content/enhanced_outputs1"

# tiling settings (robust, will auto-adjust tile if image is small)
TILE_BASE    = 768
OVERLAP_BASE = 256
MULTIPLE     = 8
USE_AMP      = True
USE_TWO_PASS = True

# postprocess toggles
DO_DEHAZE       = True
DEHAZE_STRENGTH = 0.18   # 0.12..0.22
DEHAZE_RADIUS   = 41

DO_WHITE_BALANCE = True
WB_STRENGTH      = 0.60  # 0.45..0.70

DO_CLAHE_L      = True
CLAHE_CLIP      = 1.2
CLAHE_TILEGRID  = 8
CLAHE_BLEND     = 0.25   # 0.20..0.35

DO_SHARPEN      = True
SHARP_AMOUNT    = 0.12
SHARP_RADIUS    = 1.2
SHARP_EDGE_BIAS = 0.85

os.makedirs(out_dir, exist_ok=True)

# ============================================================
# 3) Load model (ema/model/raw supported)
# ============================================================
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

# ============================================================
# 4) Helpers (robust padding, conversions, side-by-side)
# ============================================================
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def tensor_to_np_uint8(t: torch.Tensor) -> np.ndarray:
    t = t.clamp(0, 1)[0].permute(1, 2, 0).detach().cpu().numpy()
    return (t * 255.0 + 0.5).astype(np.uint8)

def safe_pad(x: torch.Tensor, pad_l, pad_r, pad_t, pad_b, mode="reflect") -> torch.Tensor:
    # reflect padding fails if pad >= input size on that dimension
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

def make_compare_strip(inp_u8, raw_u8, post_u8):
    # ensure same H
    H = min(inp_u8.shape[0], raw_u8.shape[0], post_u8.shape[0])
    def cropH(im): return im[:H, :, :]
    inp_u8, raw_u8, post_u8 = cropH(inp_u8), cropH(raw_u8), cropH(post_u8)

    # simple separator
    sep = np.zeros((H, 12, 3), dtype=np.uint8)
    return np.concatenate([inp_u8, sep, raw_u8, sep, post_u8], axis=1)

# ============================================================
# 5) Robust tiled inference (auto tile/overlap per image)
# ============================================================
@torch.no_grad()
def enhance_tiled(x: torch.Tensor, tile, overlap, multiple=8, use_amp=True, offset_y=0, offset_x=0):
    assert x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 3
    _, _, H, W = x.shape

    # pad to multiple
    xpad, _ = pad_to_multiple(x, m=multiple)
    _, _, Hm, Wm = xpad.shape

    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be < tile")

    # pad so tiling covers fully
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

    # smooth blending window
    yy = torch.linspace(0, math.pi, tile, device=x.device).view(1, 1, tile, 1)
    xx = torch.linspace(0, math.pi, tile, device=x.device).view(1, 1, 1, tile)
    win = (torch.sin(yy) * torch.sin(xx)).clamp_min(1e-6)

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and x.is_cuda))

    start_y = int(offset_y) % stride
    start_x = int(offset_x) % stride

    ys = list(range(start_y, Hp - tile + 1, stride))
    xs = list(range(start_x, Wp - tile + 1, stride))
    if 0 not in ys: ys = [0] + ys
    if 0 not in xs: xs = [0] + xs
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

    # crop back
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
    """
    Chooses tile/overlap automatically per image so it works for any size.
    """
    _, _, H, W = x.shape

    # choose tile not bigger than image (keep min 128)
    tile = min(TILE_BASE, max(128, (min(H, W) // 32) * 32))
    tile = int(tile)

    # overlap as fraction, but always < tile
    overlap = min(OVERLAP_BASE, tile // 3)
    overlap = int(overlap)

    # if image is tiny, just do single forward with safe pad-to-multiple
    if H <= tile and W <= tile:
        xpad, _ = pad_to_multiple(x, m=MULTIPLE)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(USE_AMP and x.is_cuda)):
            y = model(xpad).float().clamp(0, 1)
        return y[:, :, :H, :W]

    # tiled
    if USE_TWO_PASS:
        return enhance_two_pass(x, tile=tile, overlap=overlap)
    else:
        return enhance_tiled(x, tile=tile, overlap=overlap, multiple=MULTIPLE, use_amp=USE_AMP)

# ============================================================
# 6) Post-processing (underwater-safe)
# ============================================================
def dehaze_light(img_u8: np.ndarray, strength=0.18, radius=41) -> np.ndarray:
    rgb = img_u8.astype(np.float32) / 255.0
    dc = np.min(rgb, axis=2)
    dc = cv2.erode(dc, np.ones((7,7), np.uint8))
    t = 1.0 - float(strength) * cv2.GaussianBlur(dc, (0,0), radius/6)
    t = np.clip(t, 0.75, 1.0)

    A = np.percentile(rgb.reshape(-1,3), 99.5, axis=0)
    out = (rgb - A) / t[..., None] + A
    out = np.clip(out, 0, 1)
    return (out * 255.0 + 0.5).astype(np.uint8)

def wb_lab_chroma_safe(img_u8: np.ndarray, strength=0.60, dark_thresh=10) -> np.ndarray:
    rgb = img_u8
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    mask = (L > dark_thresh).astype(np.uint8)
    if mask.sum() < 200:
        return img_u8

    a_mean = cv2.mean(A, mask=mask)[0]
    b_mean = cv2.mean(B, mask=mask)[0]

    a_shift = float(np.clip((128.0 - a_mean) * float(strength), -12, 12))
    b_shift = float(np.clip((128.0 - b_mean) * float(strength), -12, 12))

    A2 = np.clip(A + a_shift, 0, 255)
    B2 = np.clip(B + b_shift, 0, 255)

    lab2 = cv2.merge([L, A2, B2]).astype(np.uint8)
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)

def clahe_on_luminance_blend(img_u8: np.ndarray, clip=1.2, tilegrid=8, alpha=0.25) -> np.ndarray:
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tilegrid), int(tilegrid)))
    L2 = clahe.apply(L)
    Lb = cv2.addWeighted(L, 1.0 - float(alpha), L2, float(alpha), 0.0)

    lab2 = cv2.merge([Lb, A, B])
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)

def edge_mask(img_u8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    mag = np.abs(lap)
    p95 = np.percentile(mag, 95) + 1e-6
    m = np.clip(mag / p95, 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 1.0)
    return m

def unsharp_edge_masked(img_u8: np.ndarray, amount=0.12, radius=1.2, edge_bias=0.85) -> np.ndarray:
    img = img_u8.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), float(radius))
    sharp = img + float(amount) * (img - blurred)

    m = edge_mask(img_u8)
    m2 = np.clip((m - edge_bias) / max(1e-6, (1.0 - edge_bias)), 0, 1)
    m3 = m2[..., None]

    out = img * (1.0 - m3) + sharp * m3
    return np.clip(out, 0, 255).astype(np.uint8)

def postprocess(img_u8: np.ndarray) -> np.ndarray:
    out = img_u8

    if DO_DEHAZE:
        out = dehaze_light(out, strength=DEHAZE_STRENGTH, radius=DEHAZE_RADIUS)

    if DO_WHITE_BALANCE:
        out = wb_lab_chroma_safe(out, strength=WB_STRENGTH)

    if DO_CLAHE_L:
        out = clahe_on_luminance_blend(out, clip=CLAHE_CLIP, tilegrid=CLAHE_TILEGRID, alpha=CLAHE_BLEND)

    if DO_SHARPEN:
        out = unsharp_edge_masked(out, amount=SHARP_AMOUNT, radius=SHARP_RADIUS, edge_bias=SHARP_EDGE_BIAS)

    return out

# ============================================================
# 7) Run batch: save model-only + post + side-by-side
# ============================================================
paths = sorted(glob.glob(os.path.join(inp_dir, "*.*")))
print("Found images:", len(paths))
print("Saving to:", out_dir)

for p in tqdm(paths):
    img_pil = Image.open(p).convert("RGB")
    inp_u8  = np.asarray(img_pil).astype(np.uint8)

    x = pil_to_tensor(img_pil).to(device)

    # model inference (robust tiling)
    y = run_model_robust(x)

    # outputs
    raw_u8  = tensor_to_np_uint8(y)            # model-only output
    post_u8 = postprocess(raw_u8.copy())       # postprocessed output
    comp_u8 = make_compare_strip(inp_u8, raw_u8, post_u8)

    base = os.path.splitext(os.path.basename(p))[0]
    out_raw  = os.path.join(out_dir, f"{base}_raw_model.png")
    out_post = os.path.join(out_dir, f"{base}_post.png")
    out_cmp  = os.path.join(out_dir, f"{base}_compare.png")

    Image.fromarray(raw_u8).save(out_raw)
    Image.fromarray(post_u8).save(out_post)
    Image.fromarray(comp_u8).save(out_cmp)

print("✅ Done. Check:", out_dir)
!zip -r /content/enhanced_outputs1.zip /content/enhanced_outputs1

from google.colab import files
files.download('/content/enhanced_outputs1.zip')

