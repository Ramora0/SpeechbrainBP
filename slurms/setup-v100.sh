#!/bin/bash
# One-time setup: create .v100 venv and install dependencies
# Run from the repo root: bash slurms/setup-v100.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${REPO_DIR}/.v100"
SB_DIR="${HOME}/research/speechbrain"

echo "=== Setting up .v100 venv in ${VENV_DIR} ==="

module purge
module load cuda/12.4.1

# Create venv (or reuse existing)
if [ ! -f "${VENV_DIR}/bin/python" ]; then
    echo "Creating venv..."
    python3 -m venv "${VENV_DIR}"
else
    echo "Venv already exists, reusing."
fi

source "${VENV_DIR}/bin/activate"

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing SpeechBrain from ${SB_DIR} ==="
pip install -e "${SB_DIR}"

echo "=== Installing remaining dependencies ==="
pip install "huggingface_hub<0.24" sentencepiece hyperpyyaml wandb

echo "=== Verifying ==="
python -c "import speechbrain; print(f'SpeechBrain {speechbrain.__version__} OK')"
python -c "import torch; print(f'PyTorch {torch.__version__} CUDA={torch.cuda.is_available()}')"
python -c "import sentencepiece; print('sentencepiece OK')"

echo "=== Done! Ready for: sbatch slurms/train-v100.slurm ==="
