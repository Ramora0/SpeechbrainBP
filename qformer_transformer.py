"""TransformerASR variant wired for the Qformer track.

Standalone `nn.Module` (not a subclass of `TransformerASR`) because the
upstream `forward` / `encode` hard-code a single-src assumption. This module
owns the pieces individually: custom_src_module, self-attn positional
encoding, the Qformer Conformer encoder, και a standard TransformerDecoder.

The forward pass accepts `(src, kv, tgt, query_lens, kv_lens)`:
- `src`: subsampled queries post-CNN, shape (B, S, input_size)
- `kv`:  strided kv post-CNN, shape (B, K, input_size) — already subsampled
        by `kv_stride` in the Downsampler
- Both are projected via a shared `custom_src_module` into d_model.
- Cross-attention in the first N encoder layers uses RoPE με original-frame
  positions (q_i at frame i·query_stride, k_j at frame j·kv_stride) so
  attention respects the actual temporal layout.

A matching `decode(tgt, encoder_out, enc_len)` is provided so the standard
`S2STransformerBeamSearcher` works without modification.
"""

from typing import Optional

import torch
import torch.nn as nn

from speechbrain.dataio.dataio import length_to_mask
from speechbrain.lobes.models.transformer.Transformer import (
    NormalizedEmbedding,
    PositionalEncoding,
    TransformerDecoder,
    get_key_padding_mask,
    get_lookahead_mask,
)
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import RelPosEncXL
from speechbrain.nnet.containers import ModuleList
from speechbrain.nnet.linear import Linear

from qformer_conformer import QformerConformerEncoder


class QformerTransformerASR(nn.Module):
    def __init__(
        self,
        tgt_vocab,
        input_size,
        query_stride: int,
        kv_stride: int,
        d_model=144,
        nhead=4,
        num_encoder_layers=12,
        num_decoder_layers=4,
        num_cross_attn_layers=4,
        d_ffn=1024,
        dropout=0.1,
        activation=nn.GELU,
        kernel_size: int = 31,
        bias: bool = True,
        conformer_activation: type = Swish,
        attention_type: str = "RelPosMHAXL",
        max_length: int = 4000,
        rope_max_length: int = 4000,
        causal: bool = False,
    ):
        super().__init__()
        if attention_type not in ("RelPosMHAXL", "regularMHA"):
            raise ValueError(
                f"Unsupported attention_type: {attention_type}. "
                "Use 'RelPosMHAXL' or 'regularMHA'."
            )
        self.attention_type = attention_type
        self.causal = causal
        self.query_stride = query_stride
        self.kv_stride = kv_stride

        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size, n_neurons=d_model,
                bias=True, combine_dims=False,
            ),
            nn.Dropout(dropout),
        )
        self.custom_tgt_module = ModuleList(NormalizedEmbedding(d_model, tgt_vocab))

        if attention_type == "RelPosMHAXL":
            self.positional_encoding = RelPosEncXL(d_model)
        else:
            self.positional_encoding = PositionalEncoding(d_model, max_length)
        self.positional_encoding_decoder = PositionalEncoding(d_model, max_length)

        self.encoder = QformerConformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            query_stride=query_stride,
            kv_stride=kv_stride,
            num_cross_attn_layers=num_cross_attn_layers,
            kernel_size=kernel_size,
            activation=conformer_activation,
            bias=bias,
            dropout=dropout,
            causal=causal,
            attention_type=attention_type,
            rope_max_length=rope_max_length,
        )

        # The decoder always uses regularMHA — matches upstream TransformerASR
        # (Transformer.py:242, "always use regular attention in decoder").
        # The encoder's attention_type is independent.
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            d_model=d_model,
            dropout=dropout,
            activation=activation,
            normalize_before=True,
            causal=True,
            attention_type="regularMHA",
        )

        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def _build_key_padding_mask(self, seq, rel_lens):
        abs_lens = torch.round(rel_lens * seq.size(1)).long()
        return ~length_to_mask(abs_lens, max_len=seq.size(1)).bool()

    def forward(self, src, kv, tgt, query_lens, kv_lens, pad_idx=0):
        src = self.custom_src_module(src)
        kv = self.custom_src_module(kv)

        if self.attention_type == "RelPosMHAXL":
            pos_embs_self = self.positional_encoding(src)
        else:
            src = src + self.positional_encoding(src)
            pos_embs_self = None

        # No separate positional encoding on kv here — RoPE in each xattn
        # layer applies positions tied to the original fbank-frame indices.

        src_key_padding_mask = self._build_key_padding_mask(src, query_lens)
        kv_key_padding_mask = self._build_key_padding_mask(kv, kv_lens)
        src_mask = get_lookahead_mask(src) if self.causal else None

        encoder_out = self.encoder(
            src=src,
            kv=kv,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            kv_key_padding_mask=kv_key_padding_mask,
            pos_embs_self=pos_embs_self,
        )

        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)
        tgt_mask = get_lookahead_mask(tgt)

        tgt = self.custom_tgt_module(tgt)
        tgt = tgt + self.positional_encoding_decoder(tgt)

        decoder_out, _, _ = self.decoder(
            tgt=tgt,
            memory=encoder_out,
            memory_mask=None,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=None,
            pos_embs_src=None,
        )
        return encoder_out, decoder_out

    @torch.no_grad()
    def decode(self, tgt, encoder_out, enc_len=None):
        tgt_mask = get_lookahead_mask(tgt)
        src_key_padding_mask: Optional[torch.Tensor] = None
        if enc_len is not None:
            src_key_padding_mask = (1 - length_to_mask(enc_len)).bool()

        tgt = self.custom_tgt_module(tgt)
        tgt = tgt + self.positional_encoding_decoder(tgt)

        prediction, _, multihead_attns = self.decoder(
            tgt,
            encoder_out,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=None,
            pos_embs_src=None,
        )
        return prediction, multihead_attns[-1]
