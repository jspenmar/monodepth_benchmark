import logging
from typing import MutableMapping, Optional

from matplotlib import pyplot as plt
from numpy.typing import NDArray

__all__ = ['get_logger', 'flatten_dict', 'sort_dict', 'apply_cmap']


def get_logger(name: str) -> logging.Logger:
    """Get `Logger` with specified `name`, ensuring it has only one handler (including parents)."""
    l = logging.getLogger(name)
    l.propagate = False  # Don't propagate to parents (avoid duplication)
    if not l.handlers:  # Only add handlers once (avoid duplication)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(funcName)s: %(message)s'))
        l.addHandler(h)
    return l


def flatten_dict(d: MutableMapping, /, parent: str = '', sep: str = '/'):
    return dict(_flatten_dict_gen(d, parent, sep))


def _flatten_dict_gen(d, /, parent, sep):
    for k, v in d.items():
        k_new = parent+sep+k if parent else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, k_new, sep=sep).items()
        else:
            yield k_new, v


def sort_dict(d: MutableMapping):
    """Return a dict with sorted keys."""
    return {k: d[k] for k in sorted(d)}


def apply_cmap(arr: NDArray, /,
               cmap: str = 'turbo',
               vmin: Optional[float] = None,
               vmax: Optional[float] = None) -> NDArray:
    """Apply a matplotlib colormap to an image.

    :param arr: (NDArray) (*) Array of any shape to map.
    :param cmap: (str) Matplotlib colormap name.
    :param vmin: (None|float) Minimum value to use when normalizing. If `None` use `input.min()`.
    :param vmax: (None|float) Maximum value to use when normalizing. If `None` use `input.max()`.
    :return (NDArray) (*, 3) The colormapped array, where each original value has an assigned RGB value.
    """
    vmin = arr.min() if vmin is None else vmin  # Explicit `None` check to avoid issues with 0
    vmax = arr.max() if vmax is None else vmax

    arr = arr.clip(vmin, vmax)
    arr = (arr - vmin) / (vmax - vmin + 1e-5)  # Normalize [0, 1]

    arr = plt.get_cmap(cmap)(arr)[..., :-1]  # Remove alpha
    return arr
