"""Conformer encoder variant where the first N layers cross-attend to kv.

The cross-attention sub-block uses RoPE whose rotation angles are computed
from the *original frame index* of each query and key — not the index within
their (strided) sequences. Since queries come from stride `query_stride` and
kv from stride `kv_stride` of the same full-rate CNN output, query i
represents frame `i * query_stride` and key j represents frame `j * kv_stride`.
Feeding those as RoPE positions gives the attention dot product the correct
relative-position structure across the two differently-strided sequences.

Self-attention inside each layer stays on whatever the baseline encoder uses
(`RelPosMHAXL` by default), since the query-only sequence is evenly spaced
and relative positions within it are what that module expects.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.lobes.models.transformer.Conformer import (
    ConformerEncoderLayer,
    ConvolutionModule,
)
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import (
    MultiheadAttention,
    PositionalwiseFeedForward,
    RelPosMHAXL,
)
from speechbrain.nnet.normalization import LayerNorm


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class _RoPECache(nn.Module):
    """Cached cos/sin tables for RoPE, indexable by arbitrary integer positions."""

    def __init__(self, head_dim: int, max_length: int = 4000, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        half = head_dim // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, half, dtype=torch.float32) / half)
        )
        positions = torch.arange(max_length, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)  # (max_length, half)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_length, head_dim)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_length = max_length

    def get(self, positions: torch.Tensor):
        if positions.max().item() >= self.max_length:
            raise RuntimeError(
                f"RoPE position {int(positions.max())} exceeds max_length "
                f"{self.max_length}; bump max_length in the Transformer config."
            )
        return self.cos_cached[positions], self.sin_cached[positions]


def _apply_rope(x, cos, sin):
    # x:   (B, H, L, head_dim)
    # cos: (L, head_dim), sin: (L, head_dim) — broadcast over (B, H)
    return x * cos + _rotate_half(x) * sin


class RoPECrossAttention(nn.Module):
    """Standard multi-head cross-attention με RoPE on q/k at original frame positions."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        query_stride: int,
        kv_stride: int,
        dropout: float = 0.0,
        max_length: int = 4000,
        bias: bool = True,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.query_stride = query_stride
        self.kv_stride = kv_stride

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.rope = _RoPECache(self.head_dim, max_length=max_length)

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        kv_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        B, S, _ = query.shape
        K = kv.size(1)

        q = self.q_proj(query).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, K, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, K, self.nhead, self.head_dim).transpose(1, 2)

        device = q.device
        q_positions = torch.arange(S, device=device) * self.query_stride
        k_positions = torch.arange(K, device=device) * self.kv_stride
        q_cos, q_sin = self.rope.get(q_positions)
        k_cos, k_sin = self.rope.get(k_positions)
        # Cast cos/sin to q/k dtype (matters under fp16/bf16 autocast)
        q = _apply_rope(q, q_cos.to(q.dtype), q_sin.to(q.dtype))
        k = _apply_rope(k, k_cos.to(k.dtype), k_sin.to(k.dtype))

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, S, K)

        if kv_key_padding_mask is not None:
            attn = attn.masked_fill(
                kv_key_padding_mask[:, None, None, :],
                float("-inf"),
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, S, head_dim)
        out = out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.out_proj(out)


class CrossAttentionConformerEncoderLayer(nn.Module):
    """Conformer encoder layer με an extra RoPE cross-attention sub-block.

    Block order (each με residual):
        x = x + 0.5 * FFN1(x)
        x = x + SelfAttn(norm_sa(x), pos_embs=pos_embs_self)
        x = x + CrossAttn(norm_xa(x), kv, kv_key_padding_mask)   [new, RoPE]
        x = x + Conv(x, conv_mask)
        x = norm2(x + 0.5 * FFN2(x))

    Cross-attention applies RoPE με positions `arange(S) * query_stride` on
    queries και `arange(K) * kv_stride` on keys so attention respects the
    original fbank-frame positions across the two strided sequences.
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        nhead,
        query_stride: int,
        kv_stride: int,
        kernel_size=31,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        self_attention_type="RelPosMHAXL",
        rope_max_length: int = 4000,
    ):
        super().__init__()

        if self_attention_type == "RelPosMHAXL":
            self.mha_layer = RelPosMHAXL(
                num_heads=nhead, embed_dim=d_model, dropout=dropout,
                mask_pos_future=causal,
            )
        elif self_attention_type == "regularMHA":
            self.mha_layer = MultiheadAttention(
                nhead=nhead, d_model=d_model, dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unsupported self_attention_type: {self_attention_type}"
            )
        self.self_attention_type = self_attention_type

        self.cross_mha = RoPECrossAttention(
            d_model=d_model, nhead=nhead,
            query_stride=query_stride, kv_stride=kv_stride,
            dropout=dropout, max_length=rope_max_length, bias=bias,
        )
        self.norm_xa = LayerNorm(d_model)

        self.convolution_module = ConvolutionModule(
            d_model, kernel_size, bias, activation, dropout, causal=causal,
        )

        self.ffn_module1 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn, input_size=d_model, dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )
        self.ffn_module2 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn, input_size=d_model, dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x,
        kv,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        kv_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs_self: Optional[torch.Tensor] = None,
    ):
        conv_mask = None
        if src_key_padding_mask is not None:
            conv_mask = src_key_padding_mask.unsqueeze(-1)

        x = x + 0.5 * self.ffn_module1(x)

        skip = x
        x = self.norm1(x)
        x, _ = self.mha_layer(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_self,
        )
        x = x + skip

        skip = x
        x = self.norm_xa(x)
        x = self.cross_mha(x, kv, kv_key_padding_mask=kv_key_padding_mask)
        x = x + skip

        x = x + self.convolution_module(x, conv_mask)
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x


class QformerConformerEncoder(nn.Module):
    """Conformer encoder με cross-attention in the first N layers."""

    def __init__(
        self,
        num_layers,
        d_model,
        d_ffn,
        nhead,
        query_stride: int,
        kv_stride: int,
        num_cross_attn_layers=4,
        kernel_size=31,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        attention_type="RelPosMHAXL",
        rope_max_length: int = 4000,
    ):
        super().__init__()
        if not 0 <= num_cross_attn_layers <= num_layers:
            raise ValueError(
                f"num_cross_attn_layers ({num_cross_attn_layers}) must be in "
                f"[0, num_layers={num_layers}]"
            )
        self.num_cross_attn_layers = num_cross_attn_layers
        self.attention_type = attention_type

        layers = []
        for i in range(num_layers):
            if i < num_cross_attn_layers:
                layers.append(
                    CrossAttentionConformerEncoderLayer(
                        d_model=d_model, d_ffn=d_ffn, nhead=nhead,
                        query_stride=query_stride, kv_stride=kv_stride,
                        kernel_size=kernel_size, activation=activation,
                        bias=bias, dropout=dropout, causal=causal,
                        self_attention_type=attention_type,
                        rope_max_length=rope_max_length,
                    )
                )
            else:
                layers.append(
                    ConformerEncoderLayer(
                        d_model=d_model, d_ffn=d_ffn, nhead=nhead,
                        kernel_size=kernel_size, activation=activation,
                        bias=bias, dropout=dropout, causal=causal,
                        attention_type=attention_type,
                    )
                )
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
        kv,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        kv_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs_self: Optional[torch.Tensor] = None,
    ):
        out = src
        for i, layer in enumerate(self.layers):
            if i < self.num_cross_attn_layers:
                out = layer(
                    out, kv,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    kv_key_padding_mask=kv_key_padding_mask,
                    pos_embs_self=pos_embs_self,
                )
            else:
                out, _ = layer(
                    out,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos_embs=pos_embs_self,
                )
        return self.norm(out)
