import torch

from src.registry import LOSS_REG
from src.regularizers import occlusion
from src.regularizers.occlusion import *


def test_all():
    """Check all expected symbols are imported."""
    items = {'OccReg'}
    assert set(occlusion.__all__) == items, 'Incorrect keys in `__all__`.'


def test_registry():
    """Check OcclusionReg is added to LOSS_REGISTRY."""
    assert 'disp_occ' in LOSS_REG, "Missing key from loss registry."
    assert LOSS_REG['disp_occ'] == OccReg, "Incorrect class in loss registry."


def test_occlusion_ones():
    """Test OcclusionReg when all values are 1."""
    shape = 1, 1, 100, 200
    input = torch.ones(shape)

    loss, loss_dict = OccReg(invert=False).forward(input)
    assert loss == 1., "Error in `invert=False`."
    assert not loss_dict, "Unexpected keys in `loss_dict`."

    loss, loss_dict = OccReg(invert=True).forward(input)
    assert loss == -1., "Error in `invert=True`."

    loss, loss_dict = OccReg().forward(input)
    assert loss == 1., "Error in default `invert`. Expected `False`."


def test_occlusion_rand():
    """Test OcclusionReg with a random tensor."""
    shape = 1, 1, 100, 200
    input = torch.rand(shape)
    mean = input.mean()

    loss, loss_dict = OccReg(invert=False).forward(input)
    assert loss == mean, "Error in `invert=False`."
    assert not loss_dict, "Unexpected keys in `loss_dict`."

    loss, loss_dict = OccReg(invert=True).forward(input)
    assert loss == -mean, "Error in `invert=True`."

    loss, loss_dict = OccReg().forward(input)
    assert loss == mean, "Error in default `invert`. Expected `False`."
