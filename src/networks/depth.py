from typing import Optional, Sequence, Union

import timm
import torch.nn as nn
from torch import Tensor

from src import register
from src.tools import blend_stereo
from src.typing import TensorDict
from src.utils import sort_dict
from . import DECODERS

__all__ = ['DepthNet']


MASKS = {
    'explainability': 'sigmoid',
    'uncertainty': 'relu',
    None: None,
}


@register('depth')
class DepthNet(nn.Module):
    """Depth estimation network.

    Technically we predict disparity, which is normalized to the range [0, 1].
    We predict multi-scale depth at downsampling factors [1, 2, 4, 8].

    Mask types:
        - explainability: From SfMLearner. Simple mask used to weight the photometric loss
        - uncertainty: From Klodt. Predict the photometric uncertainty for each pixel using Kendall's formulation.

    NOTE: Masking is slightly different from the original papers, where the mask is predicted by the pose network.
    We follow the formulation from Monodepth2, where the depth network predicts a different mask for each of the
    support images.

    :param enc_name: (str) `timm` encoder key (check `timm.list_models()`).
    :param pretrained: (bool) If `True`, returns an encoder pretrained on ImageNet.
    :param dec_name: (str) Custom decoder type to use.
    :param out_scales: (Sequence[int]) List of multi-scale output downsampling factor as 2**s.
    :param use_virtual_stereo: (bool) If `True`, output a disparity prediction for the stereo image.
    :param mask_name: (Sequence[str]) Optional mask to predict to weight the reconstruction loss.
    :param num_ch_mask: (Sequence[int]) Number of `supp_imgs` to predict masks for.
    :param use_stereo_blend: (bool) If `True`, run a forward pass on the horizontally flipped images and blend.
    """
    def __init__(self,
                 enc_name: str = 'resnet18',
                 pretrained: bool = True,
                 dec_name: str = 'monodepth',
                 out_scales: Union[int, Sequence[int]] = (0, 1, 2, 3),
                 mask_name: Optional[str] = None,
                 num_ch_mask: Optional[int] = None,
                 use_virtual_stereo: bool = False,
                 use_stereo_blend: bool = False):
        super().__init__()
        self.enc_name = enc_name
        self.pretrained = pretrained
        self.dec_name = dec_name
        self.out_scales = [out_scales] if isinstance(out_scales, int) else out_scales
        self.mask_name = mask_name
        self.num_ch_mask = num_ch_mask
        self.use_virtual_stereo = use_virtual_stereo
        self.use_stereo_blend = use_stereo_blend

        if self.dec_name not in DECODERS:
            raise KeyError(f'Invalid decoder key. ({self.dec_name} vs. {DECODERS.keys()}')

        if self.dec_name == 'ddvnet' and self.mask_name is not None:
            raise KeyError(f'"DDVNet" is not compatible with uncertainty mask prediction.')

        if self.mask_name not in MASKS:
            raise KeyError(f'Invalid mask key. ({self.mask_name} vs. {MASKS}')

        if self.mask_name and self.num_ch_mask <= 0:
            raise ValueError(f'Number of mask channels must be greater than zero.')

        self.encoder = timm.create_model(self.enc_name, features_only=True, pretrained=pretrained)
        self.num_ch_enc = self.encoder.feature_info.channels()
        self.enc_sc = self.encoder.feature_info.reduction()

        cls = DECODERS[self.dec_name]
        self.decoders = nn.ModuleDict({
            'disp': self._get_depth_decoder(cls),
            **({f'mask': self._get_mask_decoder(cls)} if self.mask_name else {}),
        })

    def _get_depth_decoder(self, cls):
        return cls(
            num_ch_enc=self.num_ch_enc, enc_sc=self.enc_sc,
            upsample_mode='nearest', use_skip=True,
            out_sc=self.out_scales, out_ch=1 + (2*self.use_virtual_stereo), out_act='sigmoid'
        )

    def _get_mask_decoder(self, cls):
        return cls(
            num_ch_enc=self.num_ch_enc, enc_sc=self.enc_sc,
            upsample_mode='nearest', use_skip=True,
            out_sc=self.out_scales, out_ch=self.num_ch_mask, out_act=MASKS[self.mask_name]
        )

    def _forward(self, x: Tensor):
        """Base forward over a single batch."""
        feat = self.encoder(x)
        out = {'depth_feats': feat}
        for k, dec in self.decoders.items(): out[k] = sort_dict(dec(feat))

        if self.use_virtual_stereo:
            out['disp_stereo'] = {k2: v2[:, 1:] for k2, v2 in out['disp'].items()}
            out['disp'] = {k2: v2[:, :1] for k2, v2 in out['disp'].items()}

        return out

    def forward(self, x: Tensor) -> TensorDict:
        """Depth estimation forward pass.

        :param x: (Tensor) (b, 3, h, w) Input image.
        :return: {
            depth_feats: (list(Tensor)) Depth encoder multi-scale features.
            disp: (TensorDict) (b, 1, h/2**s, w/2**s) Dict mapping from scales to sigmoid disparity predictions.

            (Optional)
            disp_stereo: (TensorDict) (b, 2, h/2**s, w/2**s) Dict mapping from scales to virtual stereo predictions.
            mask: (TensorDict) (b, n, h/2**s, w/2**s) Dict mapping from scales to photometric mask predictions.
            mask_stereo: (TensorDict) (b, n, h/2**s, w/2**s) Dict mapping from scales to virtual mask predictions.
        }
        """
        out = self._forward(x)

        if self.use_stereo_blend:
            out2 = self._forward(x.flip(dims=[-1]))

            for k, v in out2.items():
                if 'disp' not in k: continue
                out[k] = {k2: blend_stereo(out[k][k2], v2.flip(dims=[-1])) for k2, v2 in v.items()}

        return out
