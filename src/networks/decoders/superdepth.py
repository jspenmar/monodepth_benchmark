from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .utils import ACT, conv3x3, conv_block

__all__ = ['SuperdepthDecoder']


class SubPixelConv(nn.Module):
    def __init__(self, ch_in: int, up_factor: int):
        super().__init__()
        ch_out = ch_in * up_factor ** 2
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), groups=ch_in, padding=1)
        self.shuffle = nn.PixelShuffle(up_factor)
        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.conv.bias)
        self.conv.weight = nn.Parameter(self.conv.weight[::4].repeat_interleave(4, 0))

    def forward(self, x):
        return self.shuffle(self.conv(x))


class SuperdepthDecoder(nn.Module):
    """From SuperDepth (https://arxiv.org/abs/1806.01260)

    Generic convolutional decoder incorporating multi-scale predictions and skip connections.

    :param num_ch_enc: (Sequence[int]) List of channels per encoder stage.
    :param enc_sc: (Sequence[int]) List of downsampling factor per encoder stage.
    :param upsample_mode: (str) Torch upsampling mode. {'nearest', 'bilinear'...}
    :param use_skip: (bool) If `True`, add skip connections from corresponding encoder stage.
    :param out_sc: (Sequence[int]) List of multi-scale output downsampling factor as 2**s.
    :param out_ch: (int) Number of output channels.
    :param out_act: (str) Activation to apply to each output stage.
    """
    def __init__(self,
                 num_ch_enc: Sequence[int],
                 enc_sc: Sequence[int],
                 upsample_mode: str = 'nearest',
                 use_skip: bool = True,
                 out_sc: Sequence[int] = (0, 1, 2, 3),
                 out_ch: int = 1,
                 out_act: str = 'sigmoid'):
        super().__init__()
        self.num_ch_enc = num_ch_enc
        self.enc_sc = enc_sc
        self.upsample_mode = upsample_mode
        self.use_skip = use_skip
        self.out_sc = out_sc
        self.out_ch = out_ch
        self.out_act = out_act

        if self.out_act not in ACT:
            raise KeyError(f'Invalid activation key. ({self.out_act} vs. {tuple(ACT.keys())}')

        self.activation = ACT[self.out_act]
        self.num_ch_dec = [16, 32, 64, 128, 256]

        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i+1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f'upconv_{i}_{0}'] = nn.Sequential(
                conv_block(num_ch_in, num_ch_out),
                SubPixelConv(num_ch_out, up_factor=2),
                nn.ReLU(inplace=True),
            )

            num_ch_in = self.num_ch_dec[i]
            scale_factor = 2**i  # NOTE: Skip connection resolution, which is current scale upsampled x2
            if self.use_skip and scale_factor in self.enc_sc:
                idx = self.enc_sc.index(scale_factor)
                num_ch_in += self.num_ch_enc[idx]
            num_ch_out = self.num_ch_dec[i]

            self.convs[f'upconv_{i}_{1}'] = conv_block(num_ch_in, num_ch_out)

        # Create multi-scale outputs
        for i in self.out_sc:
            if i == 0:
                self.convs[f'outconv_{i}'] = nn.Sequential(
                    conv3x3(self.num_ch_dec[i], self.out_ch),
                    self.activation,
                )
            else:
                self.convs[f'outconv_{i}'] = nn.Sequential(
                    conv_block(self.num_ch_dec[i], self.out_ch),
                    SubPixelConv(self.out_ch, up_factor=2 ** i),
                    self.activation,
                )

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, feat: Sequence[Tensor]) -> dict[int, Tensor]:
        out = {}
        x = feat[-1]
        for i in range(4, -1, -1):
            x = [self.convs[f'upconv_{i}_{0}'](x)]

            sf = 2**i
            if self.use_skip and sf in self.enc_sc:
                idx = self.enc_sc.index(sf)
                x += [feat[idx]]

            x = torch.cat(x, 1)
            x = self.convs[f'upconv_{i}_{1}'](x)

            if i in self.out_sc:
                out[i] = self.convs[f'outconv_{i}'](x)

        return out
