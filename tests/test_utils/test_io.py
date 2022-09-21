import numpy as np
import torch
from PIL import Image

from src.utils import io
from src.utils.io import *


def _random_pil(shape): return Image.fromarray(np.random.randint(0, 255, size=shape, dtype=np.uint8))
def _random_np(shape): return np.random.rand(*shape).astype(np.float32)
def _random_torch(shape): return torch.rand(shape, dtype=torch.float32)


def test_all():
    """Check all expected symbols are imported."""
    items = {'readlines', 'pil2np', 'np2pil', 'write_yaml', 'load_yaml', 'load_merge_yaml'}
    assert set(io.__all__) == items, 'Incorrect keys in `__all__`.'


# -----------------------------------------
def test_pil2np():
    """Test conversion from PIL to numpy."""
    shape = (100, 200, 3)
    image = _random_pil(shape)
    out = pil2np(image)
    assert isinstance(out, np.ndarray), "Output should be a numpy array."
    assert out.dtype == np.float32, "Output should be float32."
    assert out.shape == shape, "Output should be same size as input."
    assert (out.max() <= 1) and (out.min() >= 0), "Output should be normalized [0, 1]."


# -----------------------------------------
def test_np2pil():
    """Test conversion from numpy to PIL."""
    shape = h, w, _ = (100, 200, 3)
    image = _random_np(shape)

    out = np2pil(image)
    assert isinstance(out, Image.Image), "Output should be a PIL Image."
    assert out.size == (w, h), "Output should be same size as input."

    vmax, vmin = out.getextrema()[0]
    assert (vmax <= 255) and (vmin >= 0), "Output should be [0, 255]."
