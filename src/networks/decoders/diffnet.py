from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ACT, conv3x3, conv_block

__all__ = ['DiffNetDecoder']


def upsample_block(in_ch: int, out_ch: int, upsample_mode: str = 'nearest') -> nn.Module:
    """Layer to upsample the input by a factor of 2 without skip connections."""
    return nn.Sequential(
        conv_block(in_ch, out_ch),
        nn.Upsample(scale_factor=2, mode=upsample_mode),
        conv_block(out_ch, out_ch),
    )


class ChannelAttention(nn.Module):
    """Channel Attention Module incorporating Squeeze & Exicitation.

    :param in_ch: (int) Number of input channels.
    :param ratio: (int) Channels reduction ratio in bottleneck.
    """
    def __init__(self, in_ch: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch//ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch//ratio, in_ch, bias=False)
        )

        self.init_weights()

    def init_weights(self):
        """Kaiming weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        att = self.avg_pool(x)  # (b, c, 1, 1)
        att = self.fc(att.squeeze()).sigmoid()  # (b, c)
        return x * att[..., None, None]


class AttentionBlock(nn.Module):
    """Attention Block incorporating channel attention.

    :param in_ch: (int) Number of input channels.
    :param skip_ch: (int) Number of channels in skip connection features.
    :param out_ch: (Optional[int]) Number of output channels.
    :param upsample_mode: (str) Torch upsampling mode. {'nearest', 'bilinear'...}
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: Optional[int] = None, upsample_mode: str = 'nearest'):
        super().__init__()
        self.in_ch = in_ch + skip_ch
        self.out_ch = out_ch or in_ch
        self.upsample_mode = upsample_mode

        self.layers = nn.Sequential(
            ChannelAttention(self.in_ch),
            conv3x3(self.in_ch, self.out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, x_skip):
        return self.layers(torch.cat((
            F.interpolate(x, scale_factor=2, mode=self.upsample_mode),
            x_skip,
        ), dim=1))


class DiffNetDecoder(nn.Module):
    """From DiffNet (https://arxiv.org/abs/2110.09482)

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

        self.convs = nn.ModuleDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i+1]
            num_ch_out = self.num_ch_dec[i]

            scale_factor = 2 ** i  # NOTE: Skip connection resolution, which is current scale upsampled x2
            if self.use_skip and scale_factor in self.enc_sc:
                idx = self.enc_sc.index(scale_factor)
                num_ch_skip = self.num_ch_enc[idx]
                self.convs[f'upconv_{i}'] = AttentionBlock(num_ch_in, num_ch_skip, num_ch_out, self.upsample_mode)
            else:
                self.convs[f'upconv_{i}'] = upsample_block(num_ch_in, num_ch_out, self.upsample_mode)

        # Create multi-scale outputs
        for i in range(4):
            self.convs[f'outconv_{i}'] = conv3x3(self.num_ch_dec[i], self.out_ch)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.activation = ACT[self.out_act]

    def forward(self, enc_features):
        out = {}
        x = enc_features[-1]
        for i in range(4, -1, -1):

            scale_factor = 2 ** i
            if self.use_skip and scale_factor in self.enc_sc:
                idx = self.enc_sc.index(scale_factor)
                x = self.convs[f'upconv_{i}'](x, enc_features[idx])
            else:
                x = self.convs[f'upconv_{i}'](x)

            if i in self.out_sc:
                out[i] = self.activation(self.convs[f'outconv_{i}'](x))

        return out
