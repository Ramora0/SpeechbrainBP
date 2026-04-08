# CLAUDE.md

## Project Overview

Baseline Conformer-Small ASR on LibriSpeech 960h using an unmodified SpeechBrain recipe. No custom modules — this is the upstream `conformer_small` configuration (13.3M params) used as a reference point for BoundaryPredictor experiments.

## Architecture

```
Audio (16kHz) → Fbank (80 mel) → Normalize → [SpecAugment] →
CNN (2×2 = 4x compression) → Conformer Encoder (12 layers, d=144) →
Transformer Decoder (4 layers) → CTC + Seq2Seq joint loss
```

- Decoding: CTC/Attention beam search with pretrained TransformerLM
- Loss: `0.3 * CTC + 0.7 * KLdiv`
- Tokenizer: SentencePiece unigram (5000 vocab)

## Running on the Cluster

**IMPORTANT**: All cluster commands MUST use `mcp__slurm__run_command` — never local Bash.

```bash
# Submit training job
sbatch slurms/train-v100.slurm

# With a different config
sbatch slurms/train-v100.slurm hparams/conformer_small.yaml

# Extra overrides
sbatch slurms/train-v100.slurm hparams/conformer_small.yaml --number_of_epochs=50
```

Always commit and push before submitting jobs with `git_pull=True`.

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Training script (upstream SpeechBrain recipe, unmodified) |
| `librispeech_prepare.py` | Data preparation (upstream, unmodified) |
| `hparams/conformer_small.yaml` | Model config (adapted for OSC cluster paths) |
| `slurms/train-v100.slurm` | Slurm submission script |

## Data Paths (OSC Pitzer)

- LibriSpeech audio: `/fs/scratch/PAS2836/lees_stuff/librispeechbrain/LibriSpeech/`
- Shared data manifests: `/fs/scratch/PAS2836/lees_stuff/librispeechbrain/data_manifests/`
- Results/checkpoints: `/fs/scratch/PAS2836/lees_stuff/librispeechbrain/results/`

## Key Hyperparameters

- `ctc_weight: 0.3` — **do not change without good reason**
- `d_model: 144`, `num_encoder_layers: 12`, `num_decoder_layers: 4`
- `precision: fp32` (V100)
- `max_batch_length_train: 900` (dynamic batching for V100 32GB)
- `number_of_epochs: 110`

## Expected Results

From SpeechBrain benchmarks (conformer_small, 110 epochs, 960h):
- test-clean WER: ~2.49%
- test-other WER: ~6.10%
