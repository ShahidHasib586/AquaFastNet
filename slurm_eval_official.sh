#!/usr/bin/env bash
#SBATCH --job-name=uie_eval_official
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_uie_eval_official_%j.out

set -euo pipefail
cd ~/uiework/final_uie/uie_final_project
source .venv/bin/activate

python3 -c "import uie; print('uie import OK')"

python3 scripts/validate_like_train.py \
  --csv manifests/bench_euvp_test515.csv \
  --ckpt runs/uie_fastunet_base32/best.pt \
  --base 32 \
  --device cuda \
  --use_ema \
  --num_workers 1
