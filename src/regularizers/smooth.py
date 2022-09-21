import torch
import torch.nn as nn
from kornia.filters import gaussian_blur2d
from torch import Tensor

from src import register
from src.tools import ops
from src.typing import LossData

__all__ = ['SmoothReg', 'FeatSmoothReg', 'FeatPeakReg']


def compute_grad(x: Tensor, /, use_blur: bool = False, ch_mean: bool = False) -> tuple[Tensor, Tensor]:
    """Compute absolute spatial gradients in `(x, y)` directions.

    :param x: (Tensor) (b, c, h, w) Input to compute spatial gradients for.
    :param use_blur: (bool) If `True`, apply Gaussian blurring to the input.
    :param ch_mean: (bool) If `True`, return the mean gradient across channels, i.e. `c=1`.
    :return: (Tensor, Tensor) (b, (c|1), h, w) Spatial gradients in `(x, y)`.
    """
    b, c, h, w = x.shape
    if use_blur: x = gaussian_blur2d(x, kernel_size=(3, 3), sigma=(1, 1))

    dx = (x[..., :, :-1] - x[..., :, 1:]).abs()  # (b, c, h, w-1)
    dx = torch.cat((dx, x.new_zeros((b, c, h, 1))), dim=-1)  # Pad missing column with zeros.

    dy = (x[..., :-1, :] - x[..., 1:, :]).abs()  # (b, c, h-1, w)
    dy = torch.cat((dy, x.new_zeros((b, c, 1, w))), dim=-2)  # Pad missing row with zeros.

    if ch_mean:
        dx, dy = dx.mean(dim=1, keepdim=True), dy.mean(dim=1, keepdim=True)
    return dx, dy


def compute_laplacian(x: Tensor, /, use_blur: bool = False, ch_mean: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute absolute second-order spatial gradients in (xx, yy, xy, yx) directions.

    :param x: (Tensor) (b, c, h, w) Input to compute spatial gradients for.
    :param use_blur: (bool) If `True`, apply Gaussian blurring to the input.
    :param ch_mean: (bool) If `True`, return the mean gradient across channels, i.e. `c=1`.
    :return: (Tensor, Tensor, Tensor, Tensor) (b, (c|1), h, w) Second-order spatial gradients in `(xx, yy, xy, yx)`.
    """
    dx, dy = compute_grad(x, use_blur)
    dxx, dxy = compute_grad(dx, use_blur)
    dyx, dyy = compute_grad(dy, use_blur)

    if ch_mean:
        dxx, dxy = dxx.mean(dim=1, keepdim=True), dxy.mean(dim=1, keepdim=True)
        dyx, dyy = dyx.mean(dim=1, keepdim=True), dyy.mean(dim=1, keepdim=True)
    return dxx, dyy, dxy, dyx


@register('disp_smooth')
class SmoothReg(nn.Module):
    """Class implementing a disparity smoothness regularization.

    - Base: From Garg (https://arxiv.org/abs/1603.04992).
    - Edge-aware: From Monodepth (https://arxiv.org/abs/1609.03677)
    - Edge-aware + Laplacian: From DVSO (https://arxiv.org/abs/1807.02570)

    :param use_edges: (bool) If `True`, do not penalize disparity gradients aligned with image gradients.
    :param use_laplacian: (bool) If `True`, compute second-order gradients instead of first-order.
    :param use_blur: (bool) If `True`, apply Gaussian blurring to the input.
    """
    def __init__(self, use_edges: bool = False, use_laplacian: bool = False, use_blur: bool = False) -> None:
        super().__init__()
        self.use_edges = use_edges
        self.use_laplacian = use_laplacian
        self.use_blur = use_blur

        self._fn = compute_laplacian if self.use_laplacian else compute_grad

    def forward(self, disp: Tensor, img: Tensor) -> LossData:
        """Smoothness regularization forward pass.

        :param disp: (Tensor) (b, 1, h, w) Input sigmoid disparity.
        :param img: (Tensor) (b, 3, h, w) Corresponding image.
        :return:
        """
        disp = ops.mean_normalize(disp)  # Important! Otherwise degenerates to zero.
        disp_dx, disp_dy = self._fn(disp, use_blur=self.use_blur)[:2]
        disp_grad = (disp_dx.pow(2) + disp_dy.pow(2)).clamp(min=ops.eps(disp)).sqrt()

        img_dx, img_dy = self._fn(img, use_blur=self.use_blur, ch_mean=True)[:2]
        img_grad = (img_dx.pow(2) + img_dy.pow(2)).clamp(min=ops.eps(disp)).sqrt()

        if self.use_edges:
            disp_dx *= (-img_dx).exp()
            disp_dy *= (-img_dy).exp()

        loss = disp_dx.mean() + disp_dy.mean()
        return loss, {'disp_grad': disp_grad, 'image_grad': img_grad}


@register('feat_peaky')
class FeatPeakReg(nn.Module):
    """Class implementing feature gradient peakiness regularization.
    From Feat-Depth (https://arxiv.org/abs/2007.10603).

    Objective is to learn a feature representation discriminative in smooth image regions, by encouraging
    first-order gradients.

    :param use_edges: (bool) If `True`, penalize feature gradient smoothness aligned with image gradients.
    """
    def __init__(self, use_edges: bool = False):
        super().__init__()
        self.use_edges = use_edges

    def forward(self, feat: Tensor, img: Tensor) -> LossData:
        """Feature peakiness regularization forward pass.

        :param feat: (Tensor) (b, c, h, w) Input feature maps.
        :param img: (Tensor) (b, 3, h, w) Corresponding image.
        :return:
        """
        feat_dx, feat_dy = compute_grad(feat)
        feat_grad = (feat_dx.pow(2) + feat_dy.pow(2)).clamp(min=ops.eps(feat)).sqrt()

        if self.use_edges:
            dx, dy = compute_grad(img, ch_mean=True)
            feat_dx *= (-dx).exp()
            feat_dy *= (-dy).exp()

        loss = -(feat_dx.mean() + feat_dy.mean())
        return loss, {'feat_grad': feat_grad}


@register('feat_smooth')
class FeatSmoothReg(nn.Module):
    """Class implementing second-order feature gradient smoothness regularization.
    From Feat-Depth (https://arxiv.org/abs/2007.10603).

    Objective is to learn a feature representation with smooth second-order gradients to make optimization easier.

    :param use_edges: (bool) If `True`, penalize feature gradient smoothness aligned with image gradients.
    """
    def __init__(self, use_edges: bool = False) -> None:
        super().__init__()
        self.edge_aware = use_edges

    def forward(self, feat: Tensor, img: Tensor) -> LossData:
        """Second-order feature smoothness regularization forward pass.

        :param feat: (Tensor) (b, c, h, w) Input feature maps.
        :param img: (Tensor) (b, 3, h, w) Corresponding image.
        :return:
        """
        feat_dxx, feat_dyy, feat_dxy, feat_dyx = compute_laplacian(feat)
        feat_grad = (feat_dxx.pow(2) + feat_dyy.pow(2)).clamp(min=ops.eps(feat)).sqrt()

        if self.edge_aware:
            dxx, dyy, dxy, dyx = compute_laplacian(img, ch_mean=True)
            feat_dxx *= (-dxx).exp()
            feat_dyy *= (-dyy).exp()
            feat_dxy *= (-dxy).exp()
            feat_dyx *= (-dyx).exp()

        loss = feat_dxx.mean() + feat_dyy.mean() + feat_dxy.mean() + feat_dyx.mean()
        return loss, {'feat_grad': feat_grad}
