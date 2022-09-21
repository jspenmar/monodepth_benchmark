from os import PathLike
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from numpy.typing import NDArray

__all__ = [
    'readlines',
    'pil2np', 'np2pil',
    'write_yaml', 'load_yaml', 'load_merge_yaml',
]


# TEXT
# ------------------------------------------------------------------------------
def readlines(file: PathLike, /, encoding=None) -> list[str]:
    """Read file as a list of strings."""
    with open(file, encoding=encoding) as f:
        return f.read().splitlines()
# ------------------------------------------------------------------------------


# IMAGE CONVERSION
# ------------------------------------------------------------------------------
def pil2np(img: Image, /) -> NDArray:
    """Convert PIL image [0, 255] into numpy [0, 1]."""
    return np.array(img, dtype=np.float32) / 255.  # Default is float64!


def np2pil(arr: NDArray, /) -> Image:
    """Convert numpy image [0, 1] into PIL [0, 255]."""
    return Image.fromarray((arr * 255).astype(np.uint8))
# ------------------------------------------------------------------------------


# YAML LOADING
# ------------------------------------------------------------------------------
def write_yaml(file: PathLike, data: dict, mkdir: bool = False, sort_keys: bool = False) -> None:
    """Write data to a yaml file."""
    file = Path(file).with_suffix('.yaml')
    if mkdir: file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, 'w') as f:
        yaml.dump(data, f, sort_keys=sort_keys)


def load_yaml(file: PathLike) -> dict:
    """Load a single yaml file. """
    with open(file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def load_merge_yaml(*files: PathLike) -> dict:
    """Load a list of YAML cfg and recursively merge into a single config.

    Following dictionary merging rules, the first file is the "base" config, which gets updated by the second file.
    We chain this rule for however many cfg we have, i.e. ((((1 <- 2) <- 3) <- 4) ... <- n)

    :param files: (Sequence[PathLike]) List of YAML config files to load, from "oldest" to "newest".
    :return: (dict) The merged config from all given files.
    """
    data = [load_yaml(file) for file in files]
    old = data.pop(0)  # First config becomes the default which will get overwritten
    for new in data:
        old = _merge_yaml(old, new)  # Iteratively override with new cfg

    return old


def _merge_yaml(old: dict, new: dict) -> dict:
    """Recursively merge two YAML cfg.
    Dictionaries are recursively merged. All other types simply update the current value.

    NOTE: This means that a "list of dicts" will simply be updated to whatever the new value is,
    not appended to or recursively checked!

    :param old: (dict) Base dictionary containing default keys.
    :param new: (dict) New dictionary containing keys to overwrite in `old`.
    :return: (dict) The merge config.
    """
    d = old.copy()  # Just in case...
    for k, v in new.items():
        # If `v` is an existing dict, merge recursively. Otherwise replace/add `old`.
        d[k] = _merge_yaml(d[k], v) if k in d and isinstance(v, dict) else v
    return d
# ------------------------------------------------------------------------------
