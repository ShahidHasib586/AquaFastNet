#!/usr/bin/env bash
#SBATCH --job-name=uie_s2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_uie_s2_%j.out

set -euo pipefail

cd ~/uiework/final_uie/uie_final_project
source .venv/bin/activate

echo "===== SLURM INFO ====="
echo "JobID: ${SLURM_JOB_ID:-}"
echo "Node : $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "PWD  : $(pwd)"
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

OUTDIR="runs/uie_fastunet_base32"
CKPT="${OUTDIR}/last.pt"
TOTAL_EPOCHS=80   # <-- change this to whatever final epoch you want (e.g., 60, 80, 120)

echo "OUTDIR=${OUTDIR}"
echo "CKPT=${CKPT}"
echo "TOTAL_EPOCHS=${TOTAL_EPOCHS}"

ls -lah "${OUTDIR}" || true
test -f "${CKPT}" || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 3; }

python3 train.py \
  --train_csv manifests/train_pairs_train.csv \
  --val_csv manifests/train_pairs_val.csv \
  --out "${OUTDIR}" \
  --resume "${CKPT}" \
  --epochs "${TOTAL_EPOCHS}"
