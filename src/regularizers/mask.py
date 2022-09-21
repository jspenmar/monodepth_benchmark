import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src import register
from src.typing import LossData

__all__ = ['MaskReg']


@register('disp_mask')
class MaskReg(nn.Module):
    """Class implementing photometric loss masking regularization.
    From SfM-Learner (https://arxiv.org/abs/1704.07813)

    Based on the `explainability` mask, which predicts a weighting factor for each pixel in the photometric loss.
    To avoid the degenerate solution where all pixels are ignored, this regularization pushes all values towards 1
    using binary cross-entropy.
    """
    def forward(self, x: Tensor) -> LossData:
        """Mask regularization forward pass.

        :param x: (Tensor) (*) Input sigmoid explainability mask.
        :return: {
            loss: (Tensor) (,) Computed loss.
            loss_dict: (TensorDict) {}.
        }
        """
        loss = F.binary_cross_entropy(x, torch.ones_like(x))
        return loss, {}
