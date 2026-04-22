# CLAUDE.md

## Project Overview

Conformer-Small ASR on LibriSpeech 960h. Two tracks:

1. **Baseline** — upstream SpeechBrain `conformer_small` recipe (13.3M params), used as a reference point.
2. **Downsample** — same model with a pluggable `Downsampler` module inserted between the CNN frontend and the Conformer encoder. The only concrete downsampler today is `BoundaryPredictor` (per-frame MLP boundaries + mean-pool + binomial prior loss), but the training/yaml slot is generic so new downsamplers can drop in without touching the train script.

Each track has its own train script and hparams file — see **File Layout** below.

## Architecture

Baseline:
```
Audio → Fbank → Normalize → [SpecAugment] →
CNN (4x) → Conformer Encoder (12L, d=144) →
Transformer Decoder (4L) → CTC + Seq2Seq joint loss
```

Downsample:
```
Audio → Fbank → Normalize → [SpecAugment] →
CNN (4x) → Downsampler (compress) → Conformer Encoder →
Transformer Decoder → CTC + Seq2Seq + λ · downsample_aux_loss
```

- Decoding: CTC/Attention beam search with pretrained TransformerLM
- Baseline loss: `0.3 * CTC + 0.7 * KLdiv`
- Downsample loss: baseline loss + `downsample_loss_weight * ds_out.loss`
- Tokenizer: SentencePiece unigram (5000 vocab)

## Downsampler Contract

Any downsampler is an `nn.Module` whose `forward(hidden, lengths)` returns a `DownsampleOutput` (see `downsampler.py`):

| Field | Meaning |
|-------|---------|
| `hidden` | `(B, S, D)` compressed sequence — `S <= T` |
| `lengths` | relative lengths of `hidden` (`0..1`) |
| `loss` | scalar aux loss (zero tensor if none) |
| `num_output` / `num_input` | totals across the batch, for logging `keep_rate` |
| `extra_stats` | optional `{name: value}` dict, forwarded to the train logger |

Optional: if the downsampler exposes `set_temperature(t)`, `train_downsample.py` anneals `t` linearly `1.0 → ~0` across epochs (used by BoundaryPredictor's RelaxedBernoulli).

## File Layout

| Track | Train script | Hparams | Slurm |
|-------|--------------|---------|-------|
| Baseline (default) | `train.py` | `hparams/conformer_small.yaml` | `slurms/train-v100.slurm` |
| Downsample | `train_downsample.py` | `hparams/conformer_small_downsample.yaml` | `slurms/train-downsample-v100.slurm` |

`downsampler.py` — `DownsampleOutput` NamedTuple (the contract).
`boundary_predictor.py` — concrete downsampler, ported από `../speechbrainwhisper/BoundaryPredictor4.py`. Swap the `Downsampler:` line in the yaml to plug in a different one.

## Running on the Cluster

**IMPORTANT**: All cluster commands MUST use `mcp__slurm__run_command` — never local Bash.

```bash
# Baseline
sbatch slurms/train-v100.slurm
sbatch slurms/train-v100.slurm hparams/conformer_small.yaml --number_of_epochs=50

# Downsample (currently uses BoundaryPredictor)
sbatch slurms/train-downsample-v100.slurm
sbatch slurms/train-downsample-v100.slurm hparams/conformer_small_downsample.yaml --boundary_mode=all
```

Always commit and push before submitting jobs with `git_pull=True`.

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Baseline training script (upstream recipe, unmodified) |
| `train_downsample.py` | Downsample training — CNN → Downsampler → Transformer |
| `downsampler.py` | `DownsampleOutput` NamedTuple — the Downsampler contract |
| `boundary_predictor.py` | Concrete downsampler: per-frame MLP boundaries + mean pool |
| `librispeech_prepare.py` | Data preparation (upstream, unmodified) |
| `hparams/conformer_small.yaml` | Baseline config |
| `hparams/conformer_small_downsample.yaml` | Downsample config (BP instance in `Downsampler:` slot) |
| `slurms/train-v100.slurm` | Baseline slurm script |
| `slurms/train-downsample-v100.slurm` | Downsample slurm script |

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

Downsample track (`conformer_small_downsample.yaml`):
- `downsample_loss_weight: 1.0` — generic weight για whatever aux loss the Downsampler returns

BoundaryPredictor-specific (only used when the `Downsampler:` slot is `boundary_predictor.BoundaryPredictor`):
- `boundary_predictor_prior: 0.5` — target boundary fraction → 2x compression on top of the 4x CNN (8x total); conservative default for testing
- `boundary_predictor_temp: 1.0` — RelaxedBernoulli temperature (linearly annealed to ~0 by `train_downsample.py`)
- `boundary_mode: learned` — `learned` | `all` (every position) | `alternating` (every other)

## Expected Results

From SpeechBrain benchmarks (conformer_small, 110 epochs, 960h):
- test-clean WER: ~2.49%
- test-other WER: ~6.10%
