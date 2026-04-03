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
import torch
from uie.models import FastUNetEnhancer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FastUNetEnhancer(base=args.base).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    example = torch.randn(1,3,args.h,args.w, device=device).clamp(0,1)
    ts = torch.jit.trace(model, example)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    ts.save(str(out))
    print("Saved TorchScript:", out)

if __name__ == "__main__":
    main()
