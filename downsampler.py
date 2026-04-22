"""Generic interface for sequence-downsampling modules.

A Downsampler sits between the CNN frontend and the Conformer encoder:
    (B, T, D_in)  -->  Downsampler  -->  (B, S, D_in), S <= T

Any nn.Module whose `forward(hidden, lengths)` returns a `DownsampleOutput`
can be dropped into `train_downsample.py` + `conformer_small_downsample.yaml`
as the `Downsampler` slot. BoundaryPredictor (boundary_predictor.py) is the
first concrete implementation; add new ones alongside it.
"""

from typing import Any, Dict, NamedTuple, Optional

import torch


class DownsampleOutput(NamedTuple):
    """Return value contract for every Downsampler.

    Fields
    ------
    hidden : Tensor (B, S, D)
        Compressed sequence. S must be <= input T.
    lengths : Tensor (B,)
        Relative lengths (0..1) of `hidden`, as SpeechBrain expects.
    loss : Tensor (scalar)
        Auxiliary loss to add to the ASR loss. Use a zero tensor if the
        downsampler has no auxiliary objective.
    num_output : int
        Total number of output positions across the batch (for logging the
        realized compression rate).
    num_input : int
        Total number of input positions across the batch.
    extra_stats : dict[str, Any] | None
        Optional per-step scalar stats to forward into the train logger
        (e.g. {"boundary_cv": 0.31}). Values of None are dropped by the
        train loop.
    """

    hidden: torch.Tensor
    lengths: torch.Tensor
    loss: torch.Tensor
    num_output: int
    num_input: int
    extra_stats: Optional[Dict[str, Any]] = None
