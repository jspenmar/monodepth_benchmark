import torch

from src.registry import LOSS_REG
from src.regularizers import mask
from src.regularizers.mask import *


def test_all():
    """Check all expected symbols are imported."""
    items = {'MaskReg'}
    assert set(mask.__all__) == items, 'Incorrect keys in `__all__`.'


def test_registry():
    """Check MaskReg is added to LOSS_REGISTRY."""
    assert 'disp_mask' in LOSS_REG, "Missing key from loss registry."
    assert LOSS_REG['disp_mask'] == MaskReg, "Incorrect class in loss registry."


def test_mask():
    """Test MaskReg when all values are 1."""
    shape = 1, 1, 100, 200

    loss, loss_dict = MaskReg().forward(torch.ones(shape))
    assert loss == 0, "Error in correct BCELoss."
    assert not loss_dict, "Unexpected keys in `loss_dict`."

    # NOTE: This is an artefact of how PyTorch computes the BCELoss, which is clamped to `-100` to prevent `-Inf`.
    # See (https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
    loss, loss_dict = MaskReg().forward(torch.zeros(shape))
    assert loss == 100., "Error in incorrect BCELoss."
