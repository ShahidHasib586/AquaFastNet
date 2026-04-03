#!/usr/bin/env bash
#SBATCH --job-name=uie_bench_lr3e5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_uie_bench_lr3e5_%j.out

set -euo pipefail
cd ~/uiework/final_uie/uie_final_project
source .venv/bin/activate

EXTRA_EPOCHS=20
LR="3e-5"
RESUME_CKPT="runs/uie_fastunet_base32/best.pt"
VAL_CSV="manifests/bench_euvp_test515.csv"

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

if [ ! -f "$VAL_CSV" ]; then
  echo "ERROR: missing benchmark csv: $VAL_CSV"
  exit 3
fi

TARGET_EPOCHS=$(python3 - <<PY
import torch
ck=torch.load("$RESUME_CKPT", map_location="cpu")
ep=int(ck.get("epoch",0))
print(ep + int("$EXTRA_EPOCHS"))
PY
)

echo "Resume ckpt : $RESUME_CKPT"
echo "Target epoch: $TARGET_EPOCHS"
echo "LR         : $LR"
echo "Val CSV    : $VAL_CSV"

python3 train.py \
  --train_csv manifests/train_pairs_train.csv \
  --val_csv "$VAL_CSV" \
  --out runs/uie_fastunet_base32 \
  --epochs "$TARGET_EPOCHS" \
  --batch 6 \
  --crop 384 \
  --lr "$LR" \
  --base 32 \
  --accum 2 \
  --ema_decay 0.999 \
  --resume "$RESUME_CKPT"
