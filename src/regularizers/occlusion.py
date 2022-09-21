import torch
import torch.nn as nn
from torch import Tensor

from src import register
from src.typing import LossData

__all__ = ['OccReg']


@register('disp_occ')
class OccReg(nn.Module):
    """Class implementing disparity occlusion regularization.
    From DVSO (https://arxiv.org/abs/1807.02570)

    This regularization penalizes the overall disparity in the image, encouraging the network to select background
    disparities.

    NOTE: In this case we CANNOT apply mean normalization to the input disparity. By definition, this fixes the mean of
    all elements to 1, meaning the loss is impossible to minimize.

    NOTE: The benefits of applying this regularization to purely monocular supervision are unclear,
    since the loss could simply be optimized by making all disparities smaller.

    :param invert: (bool) If `True`, encourage foreground disparities instead of background.
    """
    def __init__(self, invert: bool = False):
        super().__init__()
        self.invert = invert
        self._sign = nn.Parameter(torch.tensor(-1 if self.invert else 1), requires_grad=False)

    def forward(self, x: Tensor) -> LossData:
        """Occlusion regularization forward pass.

        :param x: (Tensor) (*) Input sigmoid disparities.
        :return: {
            loss: (Tensor) (,) Computed loss.
            loss_dict: (TensorDict) {}.
        }
        """
        loss = self._sign * x.mean()
        return loss, {}
