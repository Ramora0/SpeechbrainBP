#!/bin/bash
# Submit all conformer-small experiments to the cluster
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Pulling latest changes ==="
git pull

echo "=== Submitting 7 experiments ==="

# Baseline seeds
sbatch slurms/train-v100.slurm hparams/conformer_small.yaml \
  --seed=7775
sbatch slurms/train-v100.slurm hparams/conformer_small.yaml \
  --seed=1234
sbatch slurms/train-v100.slurm hparams/conformer_small.yaml \
  --seed=4242

# 2x batch
sbatch slurms/train-v100.slurm hparams/conformer_small.yaml \
  --seed=7775 \
  --grad_accumulation_factor=2 \
  --wandb_run_name=conformer-small-2xbatch-s7775

# 2x batch + higher LR
sbatch slurms/train-v100.slurm hparams/conformer_small.yaml \
  --seed=7775 \
  --grad_accumulation_factor=2 \
  --lr_adam=0.002 \
  --wandb_run_name=conformer-small-2xbatch-highlr-s7775

# Quarter epochs (28)
sbatch slurms/train-v100.slurm hparams/conformer_small.yaml \
  --seed=7775 \
  --number_of_epochs=28 \
  --wandb_run_name=conformer-small-28ep-s7775

# fp16
sbatch slurms/train-v100.slurm hparams/conformer_small.yaml \
  --seed=7775 \
  --precision=fp16 \
  --wandb_run_name=conformer-small-fp16-s7775

echo "=== All 7 jobs submitted ==="
