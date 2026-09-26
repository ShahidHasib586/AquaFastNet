#!/usr/bin/env python3
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from uie.models import FastUNetEnhancer
from uie.losses import ComboLoss
from uie.data import PairDataset

class EMA:
    """
    Float-only EMA. Integer buffers (e.g., BatchNorm num_batches_tracked) are copied directly.
    """
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {}
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()
            else:
                # keep non-float buffers as-is
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in msd.items():
            if torch.is_floating_point(v):
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                # copy integer buffers directly
                self.shadow[k].copy_(v.detach())

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2).clamp_min(1e-12)
    return (10.0 * torch.log10(1.0 / mse)).item()

def save_ckpt(path, model, ema, opt, sch, scaler, epoch, best, args):
    ckpt = {
        "model": model.state_dict(),
        "ema": (ema.shadow if ema is not None else None),
        "opt": opt.state_dict(),
        "sch": (sch.state_dict() if sch is not None else None),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best": best,
        "args": vars(args),
    }
    torch.save(ckpt, path)

def load_ckpt(path, model, ema, opt, sch, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if ema is not None and ckpt.get("ema") is not None:
        ema.shadow = ckpt["ema"]
    if ckpt.get("opt") is not None:
        opt.load_state_dict(ckpt["opt"])
    if sch is not None and ckpt.get("sch") is not None:
        sch.load_state_dict(ckpt["sch"])
    if ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best", -1.0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--out", default="runs/uie_fastunet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--resume", default="")
    ap.add_argument("--loss", choices=["composite", "l1", "mse"], default="composite")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_device = "cuda" if device == "cuda" else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    train_ds = PairDataset(args.train_csv, crop=args.crop, training=True)
    val_ds   = PairDataset(args.val_csv, crop=args.crop, training=False)

    train_ld = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device=="cuda"),
        drop_last=True
    )
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    model = FastUNetEnhancer(base=args.base).to(device)
    loss_fn = ComboLoss(objective=args.loss).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    scaler = GradScaler(enabled=(device=="cuda"))
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    start_ep = 0
    best = -1.0
    if args.resume and Path(args.resume).exists():
        resume_meta = torch.load(args.resume, map_location="cpu", weights_only=True)
        previous_loss = resume_meta.get("args", {}).get("loss", "composite")
        if previous_loss != args.loss:
            raise ValueError("Resume objective differs; use a separate fresh run for a loss ablation.")
        start_ep, best = load_ckpt(args.resume, model, ema, opt, sch, scaler, device)
        print(f"[RESUME] epoch={start_ep} best={best:.3f}")

    for ep in range(start_ep + 1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        pbar = tqdm(train_ld, desc=f"Train {ep}/{args.epochs}")
        for step, (x, y, _) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device_type=amp_device, enabled=(device=="cuda")):
                pred = model(x)
                loss, parts = loss_fn(pred, y)
                loss = loss / args.accum

            scaler.scale(loss).backward()

            if step % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

            pbar.set_postfix(loss=float(loss.item()*args.accum), l1=parts["l1"], ssim=parts["ssim"], perc=parts["perc"])

        sch.step()

        # validate with EMA weights if available
        model.eval()
        backup = None
        if ema is not None:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)

        vals = []
        with torch.no_grad():
            for x, y, _ in tqdm(val_ld, desc="Val", leave=False):
                x = x.to(device); y = y.to(device)
                pred = model(x).clamp(0,1)
                vals.append(psnr(pred, y))
        mean_psnr = sum(vals)/len(vals)
        print(f"[Epoch {ep}] Val PSNR={mean_psnr:.3f}")

        if backup is not None:
            model.load_state_dict(backup, strict=True)

        save_ckpt(out/"last.pt", model, ema, opt, sch, scaler, ep, best, args)
        if mean_psnr > best:
            best = mean_psnr
            save_ckpt(out/"best.pt", model, ema, opt, sch, scaler, ep, best, args)

    print("Done:", out)

if __name__ == "__main__":
    main()
