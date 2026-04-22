"""Conformer encoder variant where the first N layers cross-attend to kv.

Layers 0..num_cross_attn_layers-1 are `CrossAttentionConformerEncoderLayer`s
that add a cross-attention sub-block (queries = encoder state, kv = external
full-rate features). Remaining layers are upstream `ConformerEncoderLayer`s.

The cross-attention sub-block is zero-initialised (its out_proj weights/bias
are zeroed at init), so at step 0 the encoder is exactly equivalent to a
self-attention-only Conformer encoder — the cross-attention path is learned
in from zero, Flamingo-style (minus the tanh gate).
"""

from typing import Optional

import torch
import torch.nn as nn

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


class CrossAttentionConformerEncoderLayer(nn.Module):
    """Conformer encoder layer with an extra cross-attention sub-block.

    Block order (each with residual connection):
        x = x + 0.5 * FFN1(x)
        x = x + SelfAttn(norm_sa(x), pos_embs=pos_embs_self)
        x = x + CrossAttn(norm_xa(x), kv, kv, kv_key_padding_mask)   [new]
        x = x + Conv(x, conv_mask)
        x = norm2(x + 0.5 * FFN2(x))
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        nhead,
        kernel_size=31,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        self_attention_type="RelPosMHAXL",
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

        # Cross-attention is always plain MHA — RelPosMHAXL's relative position
        # bookkeeping assumes query and key index into the same sequence.
        self.cross_mha = MultiheadAttention(
            nhead=nhead, d_model=d_model, dropout=dropout,
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

        self._zero_init_cross_attention()

    def _zero_init_cross_attention(self):
        """Zero the cross-attention out_proj so xattn starts as a no-op."""
        nn.init.zeros_(self.cross_mha.att.out_proj.weight)
        if self.cross_mha.att.out_proj.bias is not None:
            nn.init.zeros_(self.cross_mha.att.out_proj.bias)

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
        x, _ = self.cross_mha(
            x, kv, kv,
            key_padding_mask=kv_key_padding_mask,
            pos_embs=None,
        )
        x = x + skip

        x = x + self.convolution_module(x, conv_mask)
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x


class QformerConformerEncoder(nn.Module):
    """Conformer encoder with cross-attention in the first N layers."""

    def __init__(
        self,
        num_layers,
        d_model,
        d_ffn,
        nhead,
        num_cross_attn_layers=4,
        kernel_size=31,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        attention_type="RelPosMHAXL",
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
                        kernel_size=kernel_size, activation=activation,
                        bias=bias, dropout=dropout, causal=causal,
                        self_attention_type=attention_type,
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
