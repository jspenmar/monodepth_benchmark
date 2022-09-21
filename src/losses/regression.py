from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src import register
from src.tools import ops
from src.typing import LossData

__all__ = ['RegressionLoss']


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Dense L1 loss."""
    loss = (pred - target).abs()
    return loss


def log_l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Dense Log L1 loss."""
    loss = (1 + l1_loss(pred, target)).log()
    return loss


def berhu_loss(pred: Tensor, target: Tensor, delta: float = 0.2, dynamic: bool = True) -> Tensor:
    """Dense berHu loss.

    :param pred: (Tensor) Network prediction.
    :param target: (Tensor) Ground-truth target.
    :param delta: (float) Threshold above which the loss switches from L1.
    :param dynamic: (bool) If `True`, set threshold dynamically, using `delta` as the max error percentage.
    :return: (Tensor) The computed `berhu` loss.
    """
    diff = l1_loss(pred, target)
    delta = delta if not dynamic else delta*diff.max()

    diff_delta = (diff.pow(2) + delta.pow(2)) / (2*delta + ops.eps(pred))
    loss = torch.where(diff <= delta, diff, diff_delta)
    return loss


@register(('depth_regr', 'stereo_const'))
class RegressionLoss(nn.Module):
    """Class implementing a supervised regression loss.

    NOTE: The DepthHints automask is not computed here. Instead, we rely on the `MonoDepthModule` to compute it.
    Probably not the best way of doing it, but it keeps this loss clean...

    Contributions:
        - Virtual stereo consistency: From Monodepth (https://arxiv.org/abs/1609.03677)
        - Proxy berHu regression: From Kuznietsov (https://arxiv.org/abs/1702.02706)
        - Proxy LogL1 regression: From Depth Hints (https://arxiv.org/abs/1909.09051)
        - Proxy loss automasking: From Depth Hints/Monodepth2 (https://arxiv.org/abs/1909.09051)

    :param loss_name: (str) Loss type to use. {l1, log_l1, berhu}
    :param use_automask: (bool) If `True`, use DepthHints automask based on the pred/hints errors.
    """
    def __init__(self, loss_name: str = 'berhu', use_automask: bool = False):
        super().__init__()
        self.loss_name = loss_name
        self.use_automask = use_automask

        self.criterion = {
            'l1': l1_loss,
            'log_l1': log_l1_loss,
            'berhu': berhu_loss,
        }[self.loss_name]

    def forward(self, pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> LossData:
        if mask is None: mask = torch.ones_like(target)

        err = mask * self.criterion(pred, target)
        loss = err.sum()/mask.sum()
        return loss, {'err_regr': err, 'mask_regr': mask}
