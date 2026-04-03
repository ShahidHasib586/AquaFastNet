#!/usr/bin/env bash
#SBATCH --job-name=uie_bench_sX
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_uie_bench_sX_%j.out

set -euo pipefail

cd ~/uiework/final_uie/uie_final_project
source .venv/bin/activate

echo "===== SLURM INFO ====="
echo "JobID: ${SLURM_JOB_ID:-}"
echo "Node : $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "======================"

python3 - <<'PY'
import torch, os, sys
print("cuda available:", torch.cuda.is_available())
print("cuda visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU available in this job. Exiting.", file=sys.stderr)
    sys.exit(2)
print("gpu:", torch.cuda.get_device_name(0))
PY

# Make sure benchmark CSV exists (creates if missing)
python3 - <<'PY'
import csv
from pathlib import Path

EUVP = Path("/home/shahid-ahamed.hasib/Data/Data/EUVP")
inp = EUVP/"test_samples"/"Inp"
gt  = EUVP/"test_samples"/"GTr"
out = Path("manifests/bench_euvp_test515.csv")

if out.exists():
    print("Benchmark CSV exists:", out)
    raise SystemExit(0)

assert inp.exists() and gt.exists(), "Missing EUVP test_samples/Inp or GTr"
files = sorted([p.name for p in inp.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]])
pairs = []
for fn in files:
    a = inp/fn
    b = gt/fn
    if b.exists():
        pairs.append((str(a), str(b)))

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["input","gt"])
    w.writerows(pairs)

print("Wrote benchmark CSV:", out, "pairs=", len(pairs))
PY

# Resume training (20 more epochs) using stable benchmark val
python3 train.py \
  --train_csv manifests/train_pairs_train.csv \
  --val_csv manifests/bench_euvp_test515.csv \
  --out runs/uie_fastunet_base32 \
  --epochs 97 \
  --batch 6 \
  --crop 384 \
  --lr 5e-5 \
  --base 32 \
  --accum 2 \
  --ema_decay 0.999 \
  --resume runs/uie_fastunet_base32/best.pt
