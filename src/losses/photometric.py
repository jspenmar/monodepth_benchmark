from typing import Optional

import torch.nn as nn
from torch import Tensor

from src.tools import ops

__all__ = ['DenseL1Error', 'DenseL2Error', 'SSIMError', 'PhotoError']


class DenseL1Error(nn.Module):
    """Dense L1 loss averaged over channels."""
    def forward(self, pred, target):
        return (pred - target).abs().mean(dim=1, keepdim=True)


class DenseL2Error(nn.Module):
    """Dense L2 distance."""
    def forward(self, pred, target):
        return (pred - target).pow(2).sum(dim=1, keepdim=True).clamp(min=ops.eps(pred)).sqrt()


class SSIMError(nn.Module):
    """Structural similarity error."""
    def __init__(self):
        super().__init__()
        self.pool: nn.Module = nn.AvgPool2d(kernel_size=3, stride=1)
        self.refl: nn.Module = nn.ReflectionPad2d(padding=1)

        self.eps1: float = 0.01**2
        self.eps2: float = 0.03**2

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the structural similarity error between two images.

        :param pred: (Tensor) (b, c, h, w) Predicted reconstructed images.
        :param target: (Tensor) (b, c, h, w) Target images to reconstruct.
        :return: (Tensor) (b, c, h, w) Structural similarity error.
        """
        x, y = self.refl(pred), self.refl(target)
        mu_x, mu_y = self.pool(x), self.pool(y)

        sig_x = self.pool(x**2) - mu_x**2
        sig_y = self.pool(y**2) - mu_y**2
        sig_xy = self.pool(x*y) - mu_x*mu_y

        num = (2*mu_x*mu_y + self.eps1) * (2*sig_xy + self.eps2)
        den = (mu_x**2 + mu_y**2 + self.eps1) * (sig_x + sig_y + self.eps2)

        loss = ((1 - num/den) / 2).clamp(min=0, max=1)
        return loss


class PhotoError(nn.Module):
    """Class for computing the photometric error.
    From Monodepth (https://arxiv.org/abs/1609.03677)

    The SSIMLoss can be deactivated by setting `weight_ssim=0`.
    The L1Loss can be deactivated by setting `weight_ssim=1`.
    Otherwise, the loss is a weighted combination of both.

    Attributes:
    :param weight_ssim: (float) Weight controlling the contribution of the SSIMLoss. L1 weight is `1 - ssim_weight`.
    """
    def __init__(self, weight_ssim: float = 0.85):
        super().__init__()
        if weight_ssim < 0 or weight_ssim > 1:
            raise ValueError(f'Invalid SSIM weight. ({weight_ssim} vs. [0, 1])')

        self.weight_ssim: float = weight_ssim
        self.weight_l1: float = 1 - self.weight_ssim

        self.ssim: Optional[nn.Module] = SSIMError() if self.weight_ssim > 0 else None
        self.l1: Optional[nn.Module] = DenseL1Error() if self.weight_l1 > 0 else None

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the photometric error between two images.

        :param pred: (Tensor) (b, c, h, w) Predicted reconstructed images.
        :param target: (Tensor) (b, c, h, w) Target images to reconstruct.
        :return: (Tensor) (b, 1, h, w) Photometric error.
        """
        b, _, h, w = pred.shape
        loss = pred.new_zeros((b, 1, h, w))
        if self.ssim:
            loss += self.weight_ssim * self.ssim(pred, target).mean(dim=1, keepdim=True)
        if self.l1:
            loss += self.weight_l1 * self.l1(pred, target)

        return loss
