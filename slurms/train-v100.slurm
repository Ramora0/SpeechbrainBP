#!/bin/bash
#SBATCH --job-name=conformer_bp
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --account=PAS2836
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --error=slurms/slurm-%j.err

set -euo pipefail

HPARAMS_PATH="${1:-hparams/conformer_small.yaml}"
shift 2>/dev/null || true

echo "JobID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Hparams: ${HPARAMS_PATH}"

module purge
module load cuda/12.4.1

cd "${SLURM_SUBMIT_DIR}"

mkdir -p "$TMPDIR/LibriSpeech" && rsync -a --info=progress2 /fs/scratch/PAS2836/lees_stuff/librispeechbrain/LibriSpeech/ "$TMPDIR/LibriSpeech/"

source .v100/bin/activate
python train.py "${HPARAMS_PATH}" --data_folder="$TMPDIR/LibriSpeech" "$@"
