import warnings
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import ACT, conv1x1, conv3x3, conv_block

__all__ = ['HRDepthDecoder']


class FSEBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: Optional[int] = None, upsample_mode: str = 'nearest'):
        super().__init__()
        self.in_ch = in_ch + skip_ch
        self.out_ch = out_ch or in_ch
        self.upsample_mode = upsample_mode
        self.reduction = 16

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(self.in_ch, self.in_ch//self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_ch//self.reduction, self.in_ch, bias=False)
        )

        self.conv = nn.Sequential(
            conv1x1(self.in_ch, self.out_ch, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, xs_skip: Sequence[Tensor]) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x, *xs_skip], dim=1)

        y = self.avg_pool(x).squeeze()  # (b, c)
        y = self.se(y).sigmoid()
        y = y[..., None, None].expand_as(x)  # (b, c, h, w)

        x = self.conv(x*y)
        return x


class HRDepthDecoder(nn.Module):
    """From HRDepth (https://arxiv.org/pdf/2012.07356.pdf)

    :param num_ch_enc: (Sequence[int]) List of channels per encoder stage.
    :param enc_sc: (Sequence[int]) List of downsampling factor per encoder stage.
    :param upsample_mode: (str) Torch upsampling mode. {'nearest', 'bilinear'...}
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

        if not self.use_skip:
            raise ValueError('HRDepth decoder must use skip connections.')

        if len(self.enc_sc) == 4:
            warnings.warn('HRDepth requires 5 scales, but the provided backbone has only 4. '
                          'The first scale will be duplicated and upsampled!')
            self.enc_sc = [self.enc_sc[0]//2] + self.enc_sc
            self.num_ch_enc = [self.num_ch_enc[0]] + self.num_ch_enc

        if self.out_act not in ACT:
            raise KeyError(f'Invalid activation key. ({self.out_act} vs. {tuple(ACT.keys())}')

        self.activation = ACT[self.out_act]

        # self.num_ch_dec = [4, 12, 20, 40, 80] if self.mobile_encoder else [16, 32, 64, 128, 256]
        self.num_ch_dec = [ch//2 for ch in self.num_ch_enc[1:]]
        self.num_ch_dec = [self.num_ch_dec[0] // 2] + self.num_ch_dec

        self.all_idx = ['01', '11', '21', '31', '02', '12', '22', '03', '13', '04']
        self.att_idx = ['31', '22', '13', '04']
        self.non_att_idx = ['01', '11', '21', '02', '12', '03']

        self.convs = nn.ModuleDict()
        for j in range(5):
            for i in range(5 - j):
                # upconv 0

                ch_in = self.num_ch_enc[i]
                if i == 0 and j != 0:
                    ch_in //= 2

                if i == 0 and j == 4:
                    ch_in = self.num_ch_enc[i + 1] // 2

                ch_out = ch_in // 2
                self.convs[f'{i}{j}_conv_0'] = conv_block(ch_in, ch_out)

                # 04 upconv 1, only add 04 convolution
                if i == 0 and j == 4:
                    ch_in = ch_out
                    ch_out = self.num_ch_dec[i]
                    self.convs[f'{i}{j}_conv_1'] = conv_block(ch_in, ch_out)

        # Create feature SqueezeExcitation attention blocks
        for idx in self.att_idx:
            row, col = int(idx[0]), int(idx[1])
            self.convs[f'{idx}_att'] = FSEBlock(
                in_ch=self.num_ch_enc[row+1]//2,
                skip_ch=self.num_ch_enc[row] + self.num_ch_dec[row+1] * (col-1),
                upsample_mode=self.upsample_mode,
            )

        # Create base blocks
        for idx in self.non_att_idx:
            row, col = int(idx[0]), int(idx[1])
            if col == 1:
                self.convs[f'{row+1}{col-1}_conv_1'] = conv_block(
                    in_ch=self.num_ch_enc[row+1]//2 + self.num_ch_enc[row],
                    out_ch=self.num_ch_dec[row+1]
                )
            else:
                self.convs[f'{idx}_down'] = conv1x1(
                    in_ch=self.num_ch_enc[row+1]//2 + self.num_ch_enc[row] + self.num_ch_dec[row+1] * (col-1),
                    out_ch=2*self.num_ch_dec[row+1],
                    bias=False,
                )
                self.convs[f'{row+1}{col-1}_conv_1'] = conv_block(
                    in_ch=2*self.num_ch_dec[row+1],
                    out_ch=self.num_ch_dec[row+1]
                )

        # Create multi-scale outputs
        channels = [4, 8, 24, 40] if self.mobile_encoder else self.num_ch_dec
        for i, c in enumerate(channels):
            if i in self.out_sc:
                self.convs[f'outconv_{i}'] = nn.Sequential(conv3x3(c, self.out_ch), self.activation)

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def nested_conv(self, convs: Sequence[nn.Module], x: Tensor, xs_skip: Sequence[Tensor]) -> Tensor:
        x = F.interpolate(convs[0](x), scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x, *xs_skip], dim=1)
        if len(convs) == 3:
            x = convs[2](x)

        x = convs[1](x)
        return x

    def forward(self, enc_features: Sequence[Tensor]) -> dict[int, Tensor]:
        # Duplicate and upsample first scale to fake 5 encoder stages.
        if len(enc_features) == 4:
            enc_features = [F.interpolate(enc_features[0], scale_factor=2, mode=self.upsample_mode)] + enc_features

        feat = {f'{i}0': f for i, f in enumerate(enc_features)}

        for idx in self.all_idx:
            row, col = int(idx[0]), int(idx[1])
            xs_skip = [feat[f'{row}{i}'] for i in range(col)]

            if idx in self.att_idx:
                feat[f'{idx}'] = self.convs[f'{idx}_att'](
                    self.convs[f'{row+1}{col-1}_conv_0'](feat[f'{row+1}{col-1}']),
                    xs_skip
                )

            elif idx in self.non_att_idx:
                conv = [self.convs[f'{row+1}{col-1}_conv_0'], self.convs[f'{row+1}{col-1}_conv_1']]
                if col != 1 and not self.mobile_encoder:
                    conv.append(self.convs[f'{idx}_down'])

                feat[f'{idx}'] = self.nested_conv(conv, feat[f'{row+1}{col-1}'], xs_skip)

        x = feat['04']
        x = self.convs['04_conv_0'](x)
        x = self.convs['04_conv_1'](F.interpolate(x, scale_factor=2, mode=self.upsample_mode))

        out_feat = [x, feat['04'], feat['13'], feat['22']]
        out = {i: self.convs[f'outconv_{i}'](f) for i, f in enumerate(out_feat) if i in self.out_sc}
        return out


def main():
    num_enc_ch = [64, 64, 128, 256, 512]
    enc_sc = [2, 4, 8, 16, 32]

    b, h, w = 4, 256, 512
    enc_features = [torch.rand((b, c, h//s, w//s)) for s, c in zip(enc_sc, num_enc_ch)]

    net = HRDepthDecoder(num_ch_enc=num_enc_ch, enc_sc=enc_sc, out_sc=range(4), out_ch=1)
    out = net(enc_features)

    [print(key, val.shape) for key, val in out.items()]


if __name__ == '__main__':
    main()
