"""Qformer downsampler: deterministic stride-based subsample για both streams.

Three strides, all measured in INPUT (fbank) FRAMES:
- `cnn_time_stride`: the time-stride product already applied by the CNN. The
  Downsampler's input is at rate 1/cnn_time_stride relative to fbank.
- `query_stride`:    queries live at this stride από fbank. Downsampler
  subsamples its input by `query_stride // cnn_time_stride`.
- `kv_stride`:       kv lives at this stride από fbank. Downsampler
  subsamples by `kv_stride // cnn_time_stride`. When `kv_stride ==
  cnn_time_stride` (e.g. both 2), kv is the CNN output directly και we pay
  no extra subsample cost on the kv side.

Divisibility requirement: `cnn_time_stride` must divide both `query_stride`
και `kv_stride`. This is checked at construction time.

Why push stride into the CNN? Later convs operating at already-strided rates
see a proportionally larger receptive field per output position, για
strictly cheaper compute. See CLAUDE.md (Qformer section).
"""

from typing import Any, Dict, NamedTuple, Optional

import torch
import torch.nn as nn


class QformerDownsampleOutput(NamedTuple):
    queries: torch.Tensor
    query_lengths: torch.Tensor
    kv: torch.Tensor
    kv_lengths: torch.Tensor
    loss: torch.Tensor
    num_output: int
    num_input: int
    extra_stats: Optional[Dict[str, Any]] = None


class QformerDownsampler(nn.Module):
    def __init__(
        self,
        query_stride: int = 8,
        kv_stride: int = 2,
        cnn_time_stride: int = 2,
    ):
        super().__init__()
        if query_stride < 1 or kv_stride < 1 or cnn_time_stride < 1:
            raise ValueError(
                f"All strides must be >= 1, got query={query_stride}, "
                f"kv={kv_stride}, cnn={cnn_time_stride}"
            )
        if query_stride % cnn_time_stride != 0:
            raise ValueError(
                f"cnn_time_stride ({cnn_time_stride}) must divide "
                f"query_stride ({query_stride})"
            )
        if kv_stride % cnn_time_stride != 0:
            raise ValueError(
                f"cnn_time_stride ({cnn_time_stride}) must divide "
                f"kv_stride ({kv_stride})"
            )
        if query_stride < kv_stride:
            raise ValueError(
                f"query_stride ({query_stride}) must be >= kv_stride "
                f"({kv_stride})"
            )
        self.query_stride = query_stride
        self.kv_stride = kv_stride
        self.cnn_time_stride = cnn_time_stride
        self._q_subsample = query_stride // cnn_time_stride
        self._kv_subsample = kv_stride // cnn_time_stride

    @staticmethod
    def _rel_from_abs(abs_len: torch.Tensor, total: int) -> torch.Tensor:
        return (abs_len.float() / max(total, 1)).clamp(max=1.0)

    def forward(self, hidden, lengths):
        # hidden is post-CNN, at rate 1/cnn_time_stride of fbank
        B, T_cnn, _ = hidden.shape

        queries = hidden[:, :: self._q_subsample, :]
        kv = hidden[:, :: self._kv_subsample, :] if self._kv_subsample > 1 else hidden
        S = queries.size(1)
        K = kv.size(1)

        # Input lengths are at CNN-output rate (same units as hidden's T).
        abs_cnn_len = (lengths * T_cnn).round().long().clamp(min=1, max=T_cnn)
        abs_q_len = (
            (abs_cnn_len + self._q_subsample - 1) // self._q_subsample
        ).clamp(min=1, max=S)
        abs_kv_len = (
            (abs_cnn_len + self._kv_subsample - 1) // self._kv_subsample
        ).clamp(min=1, max=K)
        query_lengths = self._rel_from_abs(abs_q_len, S)
        kv_lengths = self._rel_from_abs(abs_kv_len, K)

        return QformerDownsampleOutput(
            queries=queries,
            query_lengths=query_lengths,
            kv=kv,
            kv_lengths=kv_lengths,
            loss=torch.zeros((), device=hidden.device),
            num_output=int(abs_q_len.sum().item()),
            num_input=int(abs_cnn_len.sum().item()),
            extra_stats=None,
        )
