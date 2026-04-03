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
import argparse, shutil
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}

def is_grayscale_like(img, tol=2.0):
    b,g,r = cv2.split(img)
    return (np.mean(np.abs(r-g)) < tol) and (np.mean(np.abs(r-b)) < tol)

def is_mask_like(img, uniq_max=40):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    uniq = np.unique(gray)
    if uniq.size <= 2:
        return True
    if uniq.size <= uniq_max:
        near0 = np.mean(gray < 10)
        near255 = np.mean(gray > 245)
        if (near0 + near255) > 0.98:
            return True
    return False

def is_low_contrast(img, std_min=6.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray)) < std_min

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bad_dir", required=True)
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--uniq_max", type=int, default=40)
    ap.add_argument("--std_min", type=float, default=6.0)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    bad_dir = Path(args.bad_dir); bad_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    kept = removed = 0

    for p in tqdm(files, desc="Clean TeacherA"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            removed += 1
            continue

        bad = False
        if is_grayscale_like(img, tol=args.tol): bad = True
        if is_mask_like(img, uniq_max=args.uniq_max): bad = True
        if is_low_contrast(img, std_min=args.std_min): bad = True

        if bad:
            removed += 1
            shutil.copy2(str(p), str(bad_dir / p.name))
        else:
            kept += 1
            shutil.copy2(str(p), str(out_dir / p.name))

    print(f"Kept={kept}, Removed={removed}")
    print("Clean:", out_dir)
    print("Bad  :", bad_dir)

if __name__ == "__main__":
    main()
