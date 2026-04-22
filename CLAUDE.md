# CLAUDE.md

## Project Overview

Conformer-Small ASR on LibriSpeech 960h. Two tracks:

1. **Baseline** — upstream SpeechBrain `conformer_small` recipe (13.3M params), used as a reference point.
2. **BP** — same model with a `BoundaryPredictor` inserted between the CNN frontend and the Conformer encoder. It predicts per-frame boundaries with an MLP, mean-pools the CNN output into variable-length segments, and adds a binomial prior loss on the number of boundaries.

Each track has its own train script and hparams file — see **File Layout** below.

## Architecture

Baseline:
```
Audio → Fbank → Normalize → [SpecAugment] →
CNN (4x) → Conformer Encoder (12L, d=144) →
Transformer Decoder (4L) → CTC + Seq2Seq joint loss
```

BP:
```
Audio → Fbank → Normalize → [SpecAugment] →
CNN (4x) → BoundaryPredictor (compress via mean-pool) → Conformer Encoder →
Transformer Decoder → CTC + Seq2Seq + λ · binomial(prior) loss
```

- Decoding: CTC/Attention beam search with pretrained TransformerLM
- Baseline loss: `0.3 * CTC + 0.7 * KLdiv`
- BP loss: baseline loss + `boundary_predictor_loss_weight * bp_loss`
- Tokenizer: SentencePiece unigram (5000 vocab)

## File Layout

| Track | Train script | Hparams | Slurm |
|-------|--------------|---------|-------|
| Baseline (default) | `train.py` | `hparams/conformer_small.yaml` | `slurms/train-v100.slurm` |
| BP | `train_bp.py` | `hparams/conformer_small_bp.yaml` | `slurms/train-bp-v100.slurm` |

`boundary_predictor.py` is a standalone module (ported from `../speechbrainwhisper/BoundaryPredictor4.py`). Only `train_bp.py` imports it.

## Running on the Cluster

**IMPORTANT**: All cluster commands MUST use `mcp__slurm__run_command` — never local Bash.

```bash
# Baseline
sbatch slurms/train-v100.slurm
sbatch slurms/train-v100.slurm hparams/conformer_small.yaml --number_of_epochs=50

# BP
sbatch slurms/train-bp-v100.slurm
sbatch slurms/train-bp-v100.slurm hparams/conformer_small_bp.yaml --boundary_mode=all
```

Always commit and push before submitting jobs with `git_pull=True`.

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Baseline training script (upstream recipe, unmodified) |
| `train_bp.py` | BP training script — CNN → BoundaryPredictor → Transformer |
| `boundary_predictor.py` | Per-frame MLP boundary prediction + mean pooling |
| `librispeech_prepare.py` | Data preparation (upstream, unmodified) |
| `hparams/conformer_small.yaml` | Baseline config |
| `hparams/conformer_small_bp.yaml` | BP config (adds BoundaryPredictor + BP hparams) |
| `slurms/train-v100.slurm` | Baseline slurm script |
| `slurms/train-bp-v100.slurm` | BP slurm script |

## Data Paths (OSC Pitzer)

- LibriSpeech audio: `/fs/scratch/PAS2836/lees_stuff/librispeechbrain/LibriSpeech/`
- Shared data manifests: `/fs/scratch/PAS2836/lees_stuff/librispeechbrain/data_manifests/`
- Results/checkpoints: `/fs/scratch/PAS2836/lees_stuff/librispeechbrain/results/`

## Key Hyperparameters

Shared (both tracks):
- `ctc_weight: 0.3` — **do not change without good reason**
- `d_model: 144`, `num_encoder_layers: 12`, `num_decoder_layers: 4`
- `precision: fp32` (V100)
- `max_batch_length_train: 900` (dynamic batching for V100 32GB)
- `number_of_epochs: 110`

BP-only (`conformer_small_bp.yaml`):
- `boundary_predictor_prior: 0.5` — target boundary fraction → 2x compression on top of the 4x CNN (8x total); conservative default for testing
- `boundary_predictor_temp: 1.0` — RelaxedBernoulli temperature (linearly annealed to ~0 by `train_bp.py`)
- `boundary_predictor_loss_weight: 1.0` — weight of the binomial loss added to the ASR loss
- `boundary_mode: learned` — `learned` | `all` (every position) | `alternating` (every other)

## Expected Results

From SpeechBrain benchmarks (conformer_small, 110 epochs, 960h):
- test-clean WER: ~2.49%
- test-other WER: ~6.10%
