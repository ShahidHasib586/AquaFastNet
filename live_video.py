#!/usr/bin/env python3
import argparse, time
import cv2
import numpy as np
import torch
from uie.models import FastUNetEnhancer

@torch.no_grad()
def infer_frame(model, frame_bgr, device, max_side=960):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="0", help="0 webcam OR /path/video.mp4 OR rtsp://...")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--max_side", type=int, default=960)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FastUNetEnhancer(base=args.base).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    src = int(args.src) if args.src.isdigit() else args.src
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {args.src}")

    fps = 0.0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok: break

        out = infer_frame(model, frame, device, max_side=args.max_side)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        fps = 0.9*fps + 0.1*(1.0/max(dt,1e-6))

        vis = np.concatenate([frame, out], axis=1)
        cv2.putText(vis, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.imshow("Input | Ours", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
