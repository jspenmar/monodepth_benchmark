from typing import Optional, Sequence, Union

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from torch import Tensor

from src.utils import apply_cmap
from . import geometry as geo, ops

__all__ = ['rgb_from_disp', 'rgb_from_feat']


def _get_percentile(x: NDArray, p: int) -> float:
    """Safe percentile to handle NaNs/Inf values."""
    try: return np.percentile(x, p)
    except IndexError: return 0.0


@ops.allow_np(permute=True)
def rgb_from_disp(disp: Tensor,
                  invert: bool = False,
                  cmap: str = 'turbo',
                  vmin: float = 0,
                  vmax: Optional[Union[float, Sequence[float]]] = None) -> Tensor:
    """Convert a disparity map into an RGB colormap visualization.

    :param disp: (Tensor) (b, 1, h, w) or (h, w)
    :param invert: (bool) If `True` invert depth into disparity.
    :param cmap: (str) Matplotlib colormap name.
    :param vmin: (float) Minimum value to use when normalizing.
    :param vmax: (None|float|list) Maximum value to use when normalizing. If `None` use 95th percentile.
    :return:
    """
    if isinstance(vmin, torch.Tensor): vmin = vmin.tolist()
    if isinstance(vmax, torch.Tensor): vmax = vmax.tolist()

    n = disp.ndim
    if n == 2: disp = disp[None, None]
    if n == 3: disp = disp[None]

    if invert: disp = geo.to_inv(disp)
    disp = ops.to_numpy(disp).squeeze(-1)

    if vmax is None:
        vmax = [_get_percentile(d[d > 0], 95) for d in disp]
    elif isinstance(vmax, (int, float)) or (isinstance(vmax, torch.Tensor) and vmax.ndim == 0):
        vmax = [vmax] * disp.shape[0]
    elif len(vmax) != disp.shape[0]:
        raise ValueError(f'Non-matching vmax and disp. ({len(vmax)} vs. {disp.shape[0]})')

    rgb = torch.stack(ops.to_torch([apply_cmap(d, cmap=cmap, vmin=vmin, vmax=v) for d, v in zip(disp, vmax)]))
    if n == 2 or n == 3: rgb = rgb.squeeze(0)
    return rgb


@ops.allow_np(permute=True)
def rgb_from_feat(feat: Tensor) -> Tensor:
    """Convert dense features into an RGB image via PCA.

    NOTE: PCA is computed using all features in the batch, i.e. the representation is batch dependent.

    :param feat: (Tensor) (*b, c, h, w) Dense feature representation.
    :return: (Tensor) (*b, 3, h, w) The PCAd features.
    """
    n = feat.ndim
    if n == 3: feat = feat[None]

    b, _, h, w = feat.shape
    feat = ops.to_numpy(feat.permute(0, 2, 3, 1).flatten(0, 2))  # (b, c, h, w) -> (b, h, w, c) -> (b*h*w, c)

    proj = PCA(n_components=3).fit_transform(feat)  # (n, c) -> (n, 3)
    proj -= proj.min(0)  # Normalize per channel
    proj /= proj.max(0)

    proj = ops.to_torch(proj.reshape(b, h, w, 3))  # (b*h*w, 3) -> (b, h, w, 3) -> (b, 3, h, w)
    if n == 3: proj = proj.squeeze(0)
    return proj
