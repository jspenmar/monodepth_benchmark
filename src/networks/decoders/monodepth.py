from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.typing import TensorDict
from .utils import ACT, conv3x3, conv_block

__all__ = ['MonodepthDecoder']


class MonodepthDecoder(nn.Module):
    """From Monodepth(2) (https://arxiv.org/abs/1806.01260)

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

        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i+1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f'upconv_{i}_{0}'] = conv_block(num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            sf = 2**i  # NOTE: Skip connection resolution, which is current scale upsampled x2
            if self.use_skip and sf in self.enc_sc:
                idx = self.enc_sc.index(sf)
                num_ch_in += self.num_ch_enc[idx]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f'upconv_{i}_{1}'] = conv_block(num_ch_in, num_ch_out)

        # Create multi-scale outputs
        for i in self.out_sc:
            self.convs[f'outconv_{i}'] = conv3x3(self.num_ch_dec[i], self.out_ch)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.act = ACT[self.out_act]

    def forward(self, enc_feat: Sequence[Tensor]) -> TensorDict:
        out = {}
        x = enc_feat[-1]
        for i in range(4, -1, -1):
            x = self.convs[f'upconv_{i}_{0}'](x)
            x = [F.interpolate(x, scale_factor=2, mode=self.upsample_mode)]

            sf = 2**i
            if self.use_skip and sf in self.enc_sc:
                idx = self.enc_sc.index(sf)
                x += [enc_feat[idx]]

            x = torch.cat(x, 1)
            x = self.convs[f'upconv_{i}_{1}'](x)

            if i in self.out_sc:
                out[i] = self.act(self.convs[f'outconv_{i}'](x))

        return out
