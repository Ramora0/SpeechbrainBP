"""Non-downsampling 3-conv frontend for the Qformer track.

Three speechbrain Conv2d layers + BatchNorm + GELU, following the structural
pattern of ../SpeechbrainWhisper/hparams/conformer_8x.yaml but with every
stride configured to preserve the time axis. Stride tuples follow the
SpeechBrain convention (freq_stride, time_stride): stride[1] == 1 means the
time axis is untouched. Frequency is halved twice to land the flattened
output at the 640-d Transformer input size used by the conformer_small model.

Input:  (B, T, n_mels=80)
Output: (B, T, 640)   — time axis preserved
"""

import torch
import torch.nn as nn

from speechbrain.nnet.CNN import Conv2d


class QformerFrontEnd(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        out_channels=(64, 32, 32),
        kernel_size=(3, 3),
        strides=((2, 2), (2, 1), (1, 1)),
        dropout: float = 0.1,
    ):
        super().__init__()
        out_channels = tuple(out_channels)
        kernel_size = tuple(kernel_size)
        strides = tuple(tuple(s) for s in strides)
        if len(out_channels) != 3:
            raise ValueError(
                f"out_channels must have length 3, got {len(out_channels)}: "
                f"{out_channels!r}"
            )
        if len(strides) != 3:
            raise ValueError(
                f"strides must have length 3, got {len(strides)}: {strides!r}"
            )
        for i, s in enumerate(strides):
            if len(s) != 2:
                raise ValueError(
                    f"strides[{i}] must be (freq_stride, time_stride), "
                    f"got {s!r}"
                )

        in_channels = (1, out_channels[0], out_channels[1])
        self.conv1 = Conv2d(
            in_channels=in_channels[0], out_channels=out_channels[0],
            kernel_size=kernel_size, stride=strides[0], padding="same",
        )
        self.conv2 = Conv2d(
            in_channels=in_channels[1], out_channels=out_channels[1],
            kernel_size=kernel_size, stride=strides[1], padding="same",
        )
        self.conv3 = Conv2d(
            in_channels=in_channels[2], out_channels=out_channels[2],
            kernel_size=kernel_size, stride=strides[2], padding="same",
        )

        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.bn3 = nn.BatchNorm2d(out_channels[2])
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def _apply(self, x, conv, bn):
        # Conv2d (skip_transpose=False) takes (B, T, F, C) and returns (B, T', F', C').
        x = conv(x)
        # BatchNorm2d expects (N, C, H, W). Convert (B, T, F, C) -> (B, C, T, F).
        x = x.permute(0, 3, 1, 2).contiguous()
        x = bn(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return self.drop(self.act(x))

    def forward(self, x):
        # x: (B, T, n_mels) from Fbank. Add channel dim: (B, T, n_mels, 1).
        x = x.unsqueeze(-1)
        x = self._apply(x, self.conv1, self.bn1)
        x = self._apply(x, self.conv2, self.bn2)
        x = self._apply(x, self.conv3, self.bn3)
        b, t, f, c = x.shape
        return x.reshape(b, t, f * c)
