#!/usr/bin/env bash
#SBATCH --job-name=uie_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_uie_%j.out

set -euo pipefail
export PROJ=~/uiework/final_uie
cd "$PROJ/uie_final_project"
source .venv/bin/activate

# Stage 1 (stable)
python3 train.py \
  --train_csv manifests/train_pairs_train.csv \
  --val_csv manifests/train_pairs_val.csv \
  --out runs/uie_fastunet_base32 \
  --epochs 40 \
  --batch 8 \
  --crop 256 \
  --lr 2e-4 \
  --base 32 \
  --accum 1 \
  --ema_decay 0.999

# Stage 2 (quality)
python3 train.py \
  --train_csv manifests/train_pairs_train.csv \
  --val_csv manifests/train_pairs_val.csv \
  --out runs/uie_fastunet_base32 \
  --epochs 40 \
  --batch 6 \
  --crop 384 \
  --lr 1e-4 \
  --base 32 \
  --accum 2 \
  --ema_decay 0.999 \
  --resume runs/uie_fastunet_base32/last.pt

# Stage 3 (fine)
python3 train.py \
  --train_csv manifests/train_pairs_train.csv \
  --val_csv manifests/train_pairs_val.csv \
  --out runs/uie_fastunet_base32 \
  --epochs 20 \
  --batch 6 \
  --crop 384 \
  --lr 5e-5 \
  --base 32 \
  --accum 2 \
  --ema_decay 0.999 \
  --resume runs/uie_fastunet_base32/last.pt
