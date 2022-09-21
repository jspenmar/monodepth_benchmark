from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import ACT, conv3x3, conv_block

__all__ = ['DDVNetDecoder']


def get_discrete_bins(n: int, mode: str = 'linear') -> Tensor:
    """Get the discretized disparity value depending on number of bins and quantization mode.

    All modes assume that we are quantizing sigmoid disparity, and therefore are in range [0, 1].
    Quantization modes:
        - linear: Evenly spaces out all bins.
        - exp: Spaces bins out exponentially, providing finer detail at low disparity values, ie higher depth values.

    :param n: (int) Number of bins to use.
    :param mode: (str) Quantization mode. {linear, exp}
    :return: (Tensor) (1, n, 1, 1) Computed discrete disparity bins.
    """
    bins = torch.arange(n) / n

    if mode == 'linear':
        pass
    elif mode == 'exp':
        max_depth = Tensor(200)
        bins = torch.exp(torch.log(max_depth) * (bins - 1))
    else:
        raise ValueError(f'Invalid discretization mode. "{mode}"')

    return bins.view(1, n, 1, 1)


class SelfAttentionBlock(nn.Module):
    """Self-Attention Block.

    :param ch: (int) Number of input/output channels.
    """
    def __init__(self, ch):
        super().__init__()
        self.query_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=1, padding=0), nn.ReLU(inplace=True))
        self.key_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=1, padding=0), nn.ReLU(inplace=True))
        self.value_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=1, padding=0), nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, h, w = x.shape
        query = self.query_conv(x).flatten(-2, -1)
        key = self.key_conv(x).flatten(-2, -1).permute(0, 2, 1)  # (b, h*w, c)
        value = self.value_conv(x).flatten(-2, -1)

        att = query @ key
        out = att.softmax(dim=-1) @ value
        out = out.view(b, c, h, w)
        return out


class DDVNetDecoder(nn.Module):
    """From DDVNet (https://arxiv.org/abs/2003.13951)

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
        self.num_bins = 128
        self.bins = nn.Parameter(get_discrete_bins(self.num_bins, mode='linear'))

        self.convs = OrderedDict()
        self.convs['att'] = SelfAttentionBlock(self.num_ch_enc[-1])
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f'upconv_{i}_{0}'] = conv_block(num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            scale_factor = 2 ** i  # NOTE: Skip connection resolution, which is current scale upsampled x2
            if self.use_skip and scale_factor in self.enc_sc:
                idx = self.enc_sc.index(scale_factor)
                num_ch_in += self.num_ch_enc[idx]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f'upconv_{i}_{1}'] = conv_block(num_ch_in, num_ch_out)

        # Create multi-scale outputs
        for i in self.out_sc:
            self.convs[f'outconv_{i}'] = conv3x3(self.num_ch_dec[i], self.num_bins*self.out_ch)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.activation = ACT[self.out_act]
        self.logits = {}

    def expected_disparity(self, logits: Tensor) -> Tensor:
        """Maps discrete disparity logits into the expected weighted disparity.

        :param logits: (Tensor) (b, n, h, w) Raw unnormalized predicted probabilities.
        :return: (Tensor) (b, 1, h, w) Expected disparity map.
        """
        probs = logits.softmax(dim=1)  # (b, num_bins, h, w)
        disp = (probs * self.bins).sum(dim=1, keepdim=True)  # (b, 1, h, w)
        return disp

    def argmax_disparity(self, logits: Tensor) -> Tensor:
        idx = logits.argmax(dim=1)
        one_hot = F.one_hot(idx, self.num_bins).permute(0, 3, 1, 2)
        disp = (one_hot * self.bins).sum(dim=1, keepdim=True)
        return disp

    def forward(self, enc_features: Sequence[Tensor]) -> dict[int, Tensor]:
        out = {}
        x = self.convs['att'](enc_features[-1])
        for i in range(4, -1, -1):
            x = self.convs[f'upconv_{i}_{0}'](x)
            x = [F.interpolate(x, scale_factor=2, mode=self.upsample_mode)]

            scale_factor = 2 ** i
            if self.use_skip and scale_factor in self.enc_sc:
                idx = self.enc_sc.index(scale_factor)
                x += [enc_features[idx]]

            x = torch.cat(x, 1)
            x = self.convs[f'upconv_{i}_{1}'](x)

            if i in self.out_sc:
                logits = self.convs[f'outconv_{i}'](x)
                # out[i+len(self.out_sc)] = logits
                self.logits[i] = logits
                out[i] = torch.cat([self.expected_disparity(l) for l in logits.chunk(self.out_ch, dim=1)], dim=1)

        return out
