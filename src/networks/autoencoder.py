from typing import Sequence, Union

import timm
import torch.nn as nn
from torch import Tensor

from src import register
from . import DECODERS

__all__ = ['AutoencoderNet']

from src.typing import TensorDict


@register('autoencoder')
class AutoencoderNet(nn.Module):
    """Image autoencoder network.
    From FeatDepth (https://arxiv.org/abs/2007.10603)

    Heavily based on the Depth network with some changes:
        - Single decoder
        - Produces 3 sigmoid channels (RGB)
        - No skip connections, it's an autoencoder!

    :param enc_name: (str) `timm` encoder key (check `timm.list_models()`).
    :param pretrained: (bool) If `True`, returns an encoder pretrained on ImageNet.
    :param dec_name: (str) Custom decoder type to use.
    :param out_scales: (Sequence[int]) List of multi-scale output downsampling factor as `2**s.`
    """
    def __init__(self,
                 enc_name: str = 'resnet18',
                 pretrained: bool = True,
                 dec_name: str = 'monodepth',
                 out_scales: Union[int, Sequence[int]] = (0, 1, 2, 3)):
        super().__init__()
        self.enc_name = enc_name
        self.pretrained = pretrained
        self.dec_name = dec_name
        self.out_scales = [out_scales] if isinstance(out_scales, int) else out_scales

        if self.dec_name not in DECODERS:
            raise KeyError(f'Invalid decoder key. ({self.dec_name} vs. {DECODERS.keys()}')

        self.encoder = timm.create_model(self.enc_name, features_only=True, pretrained=pretrained)
        self.num_ch_enc = self.encoder.feature_info.channels()
        self.enc_sc = self.encoder.feature_info.reduction()

        self.decoder = DECODERS[self.dec_name](
            num_ch_enc=self.num_ch_enc, enc_sc=self.enc_sc,
            upsample_mode='nearest', use_skip=False,
            out_sc=self.out_scales, out_ch=3, out_act='sigmoid'
        )

    def forward(self, x: Tensor) -> TensorDict:
        """Image autoencoder forward pass.

        :param x: (Tensor) (b, 3, h, w) Input image.
        :return: {
            autoenc_feats: (list(Tensor)) Autoencoder encoder multi-scale features.
            autoenc_imgs: (TensorDict) (b, 1, h/2**s, w/2**s) Dict mapping from scales to image reconstructions.
        }
        """
        feat = self.encoder(x)
        out = {'autoenc_feats': feat}

        k = 'autoenc_imgs'
        out[k] = self.decoder(feat)
        out[k] = {k2: out[k][k2] for k2 in sorted(out[k])}  # Sort scales in ascending order
        return out
