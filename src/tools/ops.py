from contextlib import suppress
from functools import partial, wraps
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from src.utils import MultiLevelTimer, Timer, map_container, opt_args_deco

__all__ = [
    'get_device', 'get_latest_ckpt', 'eps',
    'freeze', 'unfreeze', 'allclose', 'num_parameters',
    'to_torch', 'to_numpy', 'op', 'allow_np',
    'to', 'detach', 'reshape', 'flatten', 'normalize',
    'standardize', 'unstandardize', 'to_gray', 'mean_normalize',
    'eye_like', 'interpolate_like', 'expand_dim',
]


# UTILS
# -----------------------------------------------------------------------------
def get_device(device: Optional[Union[str, torch.device]] = None, /) -> torch.device:
    """Create torch device from str or device. Defaults to CUDA if available."""
    if isinstance(device, torch.device): return device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def get_latest_ckpt(path: PathLike,
                    ignore: Sequence[str] = None,
                    reverse: bool = False,
                    suffix: str = '.ckpt') -> Optional[Path]:
    """Return latest or earliest checkpoint in the directory. Assumes files can be sorted in a meaningful way.

    :param path: (PathLike) Directory to search in.
    :param ignore: (Sequence[str]) Filenames to ignore, e.g. corrupted?
    :param reverse: (bool) If `True`, return earliest checkpoint.
    :param suffix: (str) Expected checkpoint file extension.
    :return: (Path) Latest checkpoint file or `None`.
    """
    path = Path(path)
    ignore = ignore or []

    # Early return if there is a `last` ckpt.
    if 'last' not in ignore and (last_file := path/('last'+suffix)).is_file():
        return last_file

    files = filter(lambda f: f.suffix == suffix and f.name not in ignore,  # Check suffix and exclude "ignored".
                   sorted(path.iterdir(), reverse=not reverse))

    file = None  # Default value if no files are found.
    with suppress(StopIteration):
        file = next(files)
    return file


def eps(x: Optional[torch.Tensor] = None, /) -> float:
    """Return the `eps` value for the given `input` dtype. (default=float32 ~= 1.19e-7)"""
    dtype = torch.float32 if x is None else x.dtype
    return torch.finfo(dtype).eps
# -----------------------------------------------------------------------------


# NETWORK UTILS
# -----------------------------------------------------------------------------
def freeze(net: nn.Module, /) -> nn.Module:
    """Fix all model parameters and prevent training."""
    for p in net.parameters(): p.requires_grad = False
    return net


def unfreeze(net: nn.Module, /) -> nn.Module:
    """Make all model parameters trainable."""
    for p in net.parameters(): p.requires_grad = True
    return net


def allclose(net1: nn.Module, net2: nn.Module, /) -> bool:
    """Check if two networks are equal."""
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        try:
            if not p1.allclose(p2): return False
        except RuntimeError:  # Non-matching parameter shapes.
            return False
    return True


def num_parameters(net: nn.Module, /) -> int:
    """Get number of trainable parameters in a network."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
# -----------------------------------------------------------------------------


# CONVERSIONS
# -----------------------------------------------------------------------------
@map_container
def to_torch(x: Any, /, permute: bool = True, device: Optional[torch.device] = None) -> Any:
    """Convert given input to torch.Tensors

    :param x: (Any) Arbitrary structure to convert to tensors (see `map_apply`).
    :param permute: (bool) If `True`, permute to PyTorch convention (b, h, w, c) -> (b, c, h, w).
    :param device: (torch.device) Device to send tensors to.
    :return: (Any) Input structure, converted to tensors.
    """
    # Classes that should be ignored
    if isinstance(x, (str, Timer, MultiLevelTimer)): return x

    # NOTE: `as_tensor` allows for free numpy conversions
    x = torch.as_tensor(x, device=device)

    if permute and x.ndim > 2:
        dim = [-1, -3, -2]  # Transpose last 3 dims as (2, 0, 1)
        dim = list(range(x.ndim - 3)) + dim  # Keep higher dimensions the same
        x = x.permute(dim)

    return x


@map_container
def to_numpy(x: Any, /, permute: bool = True) -> Any:
    """Convert given input to numpy.ndarrays.

    :param x: (Any) Arbitrary structure to convert to ndarrays (see map_apply).
    :param permute: (bool) If `True`, permute from PyTorch convention (b, c, h, w) -> (b, h, w, c).
    :return: (Any) Input structure, converted to ndarrays.
    """
    # Classes that should be ignored
    if isinstance(x, (np.ndarray, str, Timer, MultiLevelTimer)): return x

    if permute and x.ndim > 2:
        dim = [-2, -1, -3]  # Transpose last 3 dims as [1, 2, 0]
        dim = list(range(x.ndim - 3)) + dim  # Keep higher dimensions the same
        x = x.permute(dim)

    return x.detach().cpu().numpy()


@map_container
def op(_x: Any, /, *args, fn: Union[str, Callable], **kwargs) -> Any:
    """Apply a function to and arbitrary input structure. `fn` can be either a function or a method to search on `_x`.

    Example:
        >>> out = fn(input, device, op='to')  # Apply x.to(device) to each item in `x`
        >>> out = fn(input, func=torch.softmax, dim=1)  # Apply torch.softmax(x, dim=1) to each item in `x`

    :param _x: (Any) Arbitrary structure to convert to tensors (see map_apply).
    :param args: (tuple) `Args` to forward to the given `func`.
    :param fn: (str|callable) Function to apply. If given a string, it will be searched as an attribute of `_x`.
    :param kwargs: (dict) `Kwargs` to forward to the given `op`.
    :return:
    """
    if isinstance(_x, (str, Timer, MultiLevelTimer)): return _x

    if isinstance(fn, str): fn = getattr(_x, fn)  # Search as attribute of `x`, e.g. x.softmax(...).
    else:                   args = (_x, *args)    # Assume we were given a callable & add `x` to `args`.

    return fn(*args, **kwargs)


# Partials for convenience
to = partial(op, fn='to', non_blocking=True)
detach = partial(op, fn='detach')
reshape = partial(op, fn='reshape')
flatten = partial(op, fn='flatten')
normalize = partial(op, fn=F.normalize)


@opt_args_deco
def allow_np(fn: Optional[Callable], permute: bool = False) -> Callable:
    """Decorator to allow for numpy.ndarray inputs in a torch function.

    Main idea is to implement the function using torch ops and apply this decorator to also make it numpy friendly.
    Since numpy.ndarray and torch.Tensor share memory (when on CPU), there shouldn't be any overhead.

    The decorated function can have an arbitrary signature. We enforce that there should only be either np.ndarray
    or torch.Tensor inputs. All other args (int, float, str...) are left unchanged.
    """
    ann = fn.__annotations__
    for k, type in ann.items():
        if type == torch.Tensor:
            ann[k] = Union[NDArray, type]

    @wraps(fn)
    def wrapper(*args, **kwargs):
        all_args = args + tuple(kwargs.values())
        any_np = any(isinstance(arg, np.ndarray) for arg in all_args)
        any_torch = any(isinstance(arg, torch.Tensor) for arg in all_args)
        if any_torch and any_np: raise ValueError("Must pass only np.ndarray or torch.Tensor!")

        if any_np: args, kwargs = to_torch((args, kwargs), permute=permute)
        out = fn(*args, **kwargs)
        if any_np: out = to_numpy(out, permute=permute)

        return out
    return wrapper
# -----------------------------------------------------------------------------


# IMAGE CONVERSIONS
# -----------------------------------------------------------------------------
StatsRGB = tuple[float, float, float]
_mean = (0.485, 0.456, 0.406)  # ImageNet mean
_std = (0.229, 0.224, 0.225)  # ImageNet std
_coeffs = (0.299, 0.587, 0.114)  # Grayscale coefficients


@allow_np(permute=True)
def standardize(x: Tensor, /, mean: StatsRGB = _mean, std: StatsRGB = _std) -> Tensor:
    """Apply standardization. Default uses ImageNet statistics."""
    shape = [1] * (x.ndim - 3) + [3, 1, 1]
    mean = x.new_tensor(mean).view(shape)
    std = x.new_tensor(std).view(shape)
    x = (x - mean) / std
    return x


@allow_np(permute=True)
def unstandardize(x: Tensor, /, mean: StatsRGB = _mean, std: StatsRGB = _std) -> Tensor:
    """Remove standardization. Default uses ImageNet statistics."""
    shape = [1] * (x.ndim - 3) + [3, 1, 1]
    mean = x.new_tensor(mean).view(shape)
    std = x.new_tensor(std).view(shape)
    x = x*std + mean
    return x


@allow_np(permute=True)
def to_gray(x: Tensor, /, coeffs: StatsRGB = _coeffs, keepdim: bool = False) -> Tensor:
    """Convert image to grayscale."""
    shape = [1] * (x.ndim - 3) + [3, 1, 1]
    coeffs = x.new_tensor(coeffs).view(shape)
    x = (x*coeffs).sum(dim=1, keepdim=keepdim)
    return x


def mean_normalize(x: Tensor, /, dim: Union[int, Sequence[int]] = (2, 3)) -> Tensor:
    """Apply mean normalization across the specified dimensions.

    :param x: (Tensor) (*) Input tensor to normalize of any shape.
    :param dim: (int | Sequence[int]) Dimension(s) to compute the mean across.
    :return: (Tensor) (*) Mean normalized input with the same shape.
    """
    return x/x.mean(dim=dim, keepdim=True).clamp(min=eps(x))
# -----------------------------------------------------------------------------


# LIKE
# -----------------------------------------------------------------------------
def eye_like(x: Tensor, /) -> Tensor:
    """Create an Identity matrix of the same dtype and size as the input.

    NOTE: The input can be of any shape, expect the final two dimensions, which must be square.

    :param x: (Tensor) (*, n, n) Input reference tensor, where `*` can be any size (including zero).
    :return: (Tensor) (*, n, n) Identity matrix with the same dtype and size as the input.
    """
    ndim = x.ndim
    if ndim < 2: raise ValueError(f'Input must have at least two dimensions! Got "{ndim}"')

    n, n2 = x.shape[-2], x.shape[-1]
    if n != n2: raise ValueError(f'Input last two dimensions must be square (*, n, n)! Got "{x.shape}"')

    view = [1]*(ndim-2) + [n, n]  # (*, n, n)
    I = torch.eye(n, dtype=x.dtype, device=x.device).view(view).expand_as(x).clone()
    return I


def interpolate_like(input: Tensor, /, other: Tensor, mode: str = 'nearest', align_corners: bool = False) -> Tensor:
    """Interpolate to match the size of `other` tensor."""
    if mode == 'nearest': align_corners = None
    return F.interpolate(input, size=other.shape[-2:], mode=mode, align_corners=align_corners)


def expand_dim(x: Tensor, /, num: Union[int, Sequence[int]], dim: Union[int, Sequence[int]] = 0, insert: bool = False) -> Tensor:
    """Expand the specified input tensor dimensions, inserting new ones if required.

    >>> expand_dim(torch.rand(1, 1, 1), num=5, dim=1, insert=False)             # (1, 1, 1) -> (1, 5, 1)
    >>> expand_dim(torch.rand(1, 1, 1), num=5, dim=1, insert=True)              # (1, 1, 1) -> (1, 5, 1, 1)
    >>> expand_dim(torch.rand(1, 1, 1), num=(5, 3), dim=(0, 1), insert=False)   # (1, 1, 1) -> (5, 3, 1)
    >>> expand_dim(torch.rand(1, 1, 1), num=(5, 3), dim=(0, 1), insert=True)    # (1, 1, 1) -> (5, 3, 1, 1, 1)

    :param x: (Tensor) (*) Input tensor of any shape.
    :param num: (int|Sequence[int]) Expansion amount for the target dimension(s).
    :param dim: (int|Sequence[int]) Dimension(s) to expand.
    :param insert: (bool) If `True`, insert a new dimension at the specified location(s).
    :return: (Tensor) (*, num, *) Expanded tensor at the given location(s).
    """
    if isinstance(num, int):
        if isinstance(dim, int): num, dim = [num], [dim]  # (1, 1) -> ([1], [1])
        else:                    num = [num]*len(dim)   # (1, [1, 2]) -> ([1, 1], [1, 2])
    elif len(num) != len(dim): raise ValueError(f'Non-matching expansion and dims. ({len(num)} vs. {len(dim)})')

    # Add new dims to expand.
    for d in (dim if insert else ()): x = x.unsqueeze(d)

    # Create target shape, leaving other dims unchanged (-1).
    sizes = [-1]*x.ndim
    for n, d in zip(num, dim): sizes[d] = n

    return x.expand(sizes)
# -----------------------------------------------------------------------------
