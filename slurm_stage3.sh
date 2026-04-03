#!/usr/bin/env bash
#SBATCH --job-name=uie_s3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_uie_s3_%j.out

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

# Stage-3: total epochs = 110 (continues from epoch 80)
python3 train.py \
  --train_csv manifests/train_pairs_train.csv \
  --val_csv manifests/train_pairs_val.csv \
  --out runs/uie_fastunet_base32 \
  --epochs 110 \
  --batch 4 \
  --crop 512 \
  --lr 5e-5 \
  --base 32 \
  --accum 3 \
  --ema_decay 0.999 \
  --resume runs/uie_fastunet_base32/last.pt
