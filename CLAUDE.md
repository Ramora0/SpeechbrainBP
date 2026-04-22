# CLAUDE.md

## Project Overview

Conformer-Small ASR on LibriSpeech 960h. Three tracks:

1. **Baseline** — upstream SpeechBrain `conformer_small` recipe (13.3M params), used as a reference point.
2. **Downsample** — same model with a pluggable `Downsampler` module inserted between the CNN frontend and the Conformer encoder. The only concrete downsampler today is `BoundaryPredictor` (per-frame MLP boundaries + mean-pool + binomial prior loss), but the training/yaml slot is generic so new downsamplers can drop in without touching the train script.
3. **Qformer** — 3-conv stride-preserving frontend + deterministic stride-8 subsample; the first 4 Conformer encoder layers cross-attend to the full-rate (un-subsampled) feature sequence. Completely isolated from Track 2 — own CNN, Downsampler, Transformer, train script, yaml.

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

Qformer:
```
Audio → Fbank → Normalize → [SpecAugment] →
QformerFrontEnd (3 convs; first strides time by cnn_time_stride, rest stride-1) →
QformerDownsampler (queries at fbank-stride query_stride; kv at fbank-stride kv_stride) →
QformerConformerEncoder (layers 1..4 cross-attend to kv with RoPE at original
                         fbank-frame positions; layers 5..12 self-attn only) →
Transformer Decoder → CTC + Seq2Seq (no aux downsampler loss)
```
At defaults (8/2/2): kv IS the CNN output (no extra subsample); queries are every 4th CNN output. Later convs see ~11-frame input RF per output (vs 7 if the CNN were stride-1 throughout), for strictly less compute.

- Decoding: CTC/Attention beam search with pretrained TransformerLM
- Baseline loss: `0.3 * CTC + 0.7 * KLdiv`
- Downsample loss: baseline loss + `downsample_loss_weight * ds_out.loss`
- Qformer loss: baseline loss only (`downsample_loss_weight = 0`)
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

The Qformer track does NOT use this contract — it needs to hand both queries AND the full-rate kv sequence to the encoder, and defines its own `QformerDownsampleOutput` in `qformer_downsampler.py`. That output is consumed by `train_qformer.py` + `qformer_transformer.QformerTransformerASR`, not by `train_downsample.py`.

## File Layout

| Track | Train script | Hparams | Slurm |
|-------|--------------|---------|-------|
| Baseline (default) | `train.py` | `hparams/conformer_small.yaml` | `slurms/train-v100.slurm` |
| Downsample | `train_downsample.py` | `hparams/conformer_small_downsample.yaml` | `slurms/train-downsample-v100.slurm` |
| Qformer | `train_qformer.py` | `hparams/conformer_small_qformer.yaml` | `slurms/train-qformer-v100.slurm` |

`downsampler.py` — `DownsampleOutput` NamedTuple (the contract).
`boundary_predictor.py` — concrete downsampler, ported από `../speechbrainwhisper/BoundaryPredictor4.py`. Swap the `Downsampler:` line in the yaml to plug in a different one.
`qformer_frontend.py` / `qformer_downsampler.py` / `qformer_conformer.py` / `qformer_transformer.py` — self-contained Qformer-track modules (see Architecture).

## Running on the Cluster

**IMPORTANT**: All cluster commands MUST use `mcp__slurm__run_command` — never local Bash.

```bash
# Baseline
sbatch slurms/train-v100.slurm
sbatch slurms/train-v100.slurm hparams/conformer_small.yaml --number_of_epochs=50

# Downsample (currently uses BoundaryPredictor)
sbatch slurms/train-downsample-v100.slurm
sbatch slurms/train-downsample-v100.slurm hparams/conformer_small_downsample.yaml --boundary_mode=all

# Qformer (8x subsample + cross-attention to full-rate kv in first 4 layers)
sbatch slurms/train-qformer-v100.slurm
sbatch slurms/train-qformer-v100.slurm hparams/conformer_small_qformer.yaml --num_cross_attn_layers=2
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
| `hparams/conformer_small_qformer.yaml` | Qformer config (stride-8 + xattn in first 4 layers) |
| `slurms/train-v100.slurm` | Baseline slurm script |
| `slurms/train-downsample-v100.slurm` | Downsample slurm script |
| `slurms/train-qformer-v100.slurm` | Qformer slurm script |
| `train_qformer.py` | Qformer training script (CNN → queries+kv → xattn encoder) |
| `qformer_frontend.py` | 3-conv stride-preserving frontend (T preserved, F halved twice → 640) |
| `qformer_downsampler.py` | `QformerDownsampler` + `QformerDownsampleOutput` (stride-8 subsample; kv passthrough) |
| `qformer_conformer.py` | `CrossAttentionConformerEncoderLayer` + `QformerConformerEncoder` |
| `qformer_transformer.py` | `QformerTransformerASR` — src+kv forward, standard decoder |

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

Qformer track (`conformer_small_qformer.yaml`):
- `query_stride: 8` — queries live at every 8th fbank frame (80ms at 100fps)
- `kv_stride: 2`   — kv lives at every 2nd fbank frame; queries then cross-attend over 4x as many kv positions as their own length
- `cnn_time_stride: 2` — how much of the time downsampling the CNN's first conv does (the CNN produces at rate 1/cnn_time_stride). Must divide both query_stride και kv_stride. Pushing stride into the CNN gives later convs a larger per-frame RF για less compute.
- `num_cross_attn_layers: 4` — how many of the 12 Conformer layers get cross-attention; remaining are plain self-attn
- `downsample_loss_weight: 0.0` — Qformer downsampler returns no aux loss; kept at 0
- `max_batch_length_train: 750` — halved από the BP track (xattn keeps a longer kv sequence around than plain self-attn)

Cross-attention uses a RoPE-based MHA whose rotation positions are computed από the ORIGINAL fbank-frame index of each query/key — so `q_i` rotates at position `i·query_stride` και `k_j` rotates at position `j·kv_stride`. This is the natural primitive when queries και keys are different-rate samplings of the same underlying timeline.

The Qformer CNN strides follow SpeechbrainWhisper's `(freq_stride, time_stride)` tuple convention — e.g. `stride=(2,2)` reduces both, `stride=(2,1)` reduces frequency only, mirroring SBW's `conformer_Nx` idiom.

## Expected Results

From SpeechBrain benchmarks (conformer_small, 110 epochs, 960h):
- test-clean WER: ~2.49%
- test-other WER: ~6.10%
