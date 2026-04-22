"""Qformer downsampler: deterministic stride-based subsample for both streams.

Two independent strides are configurable:
- `query_stride`: queries = hidden[:, ::query_stride, :]  — the sequence the
  main encoder processes (default 8, giving 8x total time compression from
  fbank when paired with the stride-preserving QformerFrontEnd).
- `kv_stride`:   kv      = hidden[:, ::kv_stride, :]      — the sequence the
  first few encoder layers cross-attend to (default 2, a 2x-compressed view
  of the full-rate CNN output; cuts xattn memory ~4x vs. kv_stride=1 while
  keeping kv substantially richer than the queries).

Unlike `downsampler.DownsampleOutput`, this module exposes both streams so
the main encoder's cross-attention layers can consume them. A separate
output type is defined here rather than extending the existing contract.
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
    def __init__(self, query_stride: int = 8, kv_stride: int = 2):
        super().__init__()
        if query_stride < 1:
            raise ValueError(f"query_stride must be >= 1, got {query_stride}")
        if kv_stride < 1:
            raise ValueError(f"kv_stride must be >= 1, got {kv_stride}")
        self.query_stride = query_stride
        self.kv_stride = kv_stride

    @staticmethod
    def _rel_from_abs(abs_len: torch.Tensor, total: int) -> torch.Tensor:
        return (abs_len.float() / max(total, 1)).clamp(max=1.0)

    def forward(self, hidden, lengths):
        B, T, _ = hidden.shape

        queries = hidden[:, :: self.query_stride, :]
        kv = hidden[:, :: self.kv_stride, :]
        S = queries.size(1)
        K = kv.size(1)

        abs_full_len = (lengths * T).round().long().clamp(min=1, max=T)
        abs_q_len = ((abs_full_len + self.query_stride - 1) // self.query_stride).clamp(min=1, max=S)
        abs_kv_len = ((abs_full_len + self.kv_stride - 1) // self.kv_stride).clamp(min=1, max=K)
        query_lengths = self._rel_from_abs(abs_q_len, S)
        kv_lengths = self._rel_from_abs(abs_kv_len, K)

        return QformerDownsampleOutput(
            queries=queries,
            query_lengths=query_lengths,
            kv=kv,
            kv_lengths=kv_lengths,
            loss=torch.zeros((), device=hidden.device),
            num_output=int(abs_q_len.sum().item()),
            num_input=int(abs_full_len.sum().item()),
            extra_stats=None,
        )
