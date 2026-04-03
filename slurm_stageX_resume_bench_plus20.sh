#!/usr/bin/env bash
#SBATCH --job-name=uie_bench_p20
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_uie_bench_p20_%j.out

set -euo pipefail
cd ~/uiework/final_uie/uie_final_project
source .venv/bin/activate

EXTRA_EPOCHS=20
RESUME_CKPT="runs/uie_fastunet_base32/last.pt"

TARGET_EPOCHS=$(python3 - <<PY
import torch
ck=torch.load("$RESUME_CKPT", map_location="cpu")
ep=int(ck.get("epoch",0))
print(ep + int("$EXTRA_EPOCHS"))
PY
)

echo "Resume ckpt: $RESUME_CKPT"
echo "Extra epochs: $EXTRA_EPOCHS"
echo "Target epochs: $TARGET_EPOCHS"

python3 train.py \
  --train_csv manifests/train_pairs_train.csv \
  --val_csv manifests/bench_euvp_test515.csv \
  --out runs/uie_fastunet_base32 \
  --epochs "$TARGET_EPOCHS" \
  --batch 6 \
  --crop 384 \
  --lr 5e-5 \
  --base 32 \
  --accum 2 \
  --ema_decay 0.999 \
  --resume "$RESUME_CKPT"
