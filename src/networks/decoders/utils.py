from collections import OrderedDict

import torch.nn as nn

ACT = {
    'sigmoid': nn.Sigmoid(),
    'relu': nn.ReLU(inplace=True),
    'none': nn.Identity(),
    None: nn.Identity(),
}


def conv1x1(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
    """Layer to convolve input."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), bias=bias)


def conv3x3(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
    """Layer to pad and convolve input."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), padding=1, padding_mode='reflect', bias=bias)


def conv_block(in_ch: int, out_ch: int) -> nn.Module:
    """Layer to perform a convolution followed by ELU."""
    return nn.Sequential(OrderedDict({
        'conv': conv3x3(in_ch, out_ch),
        'act': nn.ELU(inplace=True),
    }))
