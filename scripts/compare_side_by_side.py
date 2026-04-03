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
from tqdm import tqdm


def load_rgb(p: Path):
    return Image.open(p).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--enhanced_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    enhanced_dir = Path(args.enhanced_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted([p for p in raw_dir.iterdir() if p.suffix.lower() in exts])

    print("Found", len(files), "raw images")

    for raw_path in tqdm(files, desc="Comparing"):
        enhanced_path = enhanced_dir / raw_path.name
        if not enhanced_path.exists():
            continue

        raw_img = load_rgb(raw_path)
        enh_img = load_rgb(enhanced_path)

        # Resize enhanced if needed (safety)
        if raw_img.size != enh_img.size:
            enh_img = enh_img.resize(raw_img.size)

        W, H = raw_img.size

        canvas = Image.new("RGB", (W * 2, H))
        canvas.paste(raw_img, (0, 0))
        canvas.paste(enh_img, (W, 0))

        canvas.save(out_dir / raw_path.name)

    print("Saved comparisons to:", out_dir)


if __name__ == "__main__":
    main()
