
from src.registry import LOSS_REG
from src.regularizers import smooth
from src.regularizers.smooth import *


def test_all():
    """Check all expected symbols are imported."""
    items = {'SmoothReg', 'FeatPeakReg', 'FeatSmoothReg'}
    assert set(smooth.__all__) == items, 'Incorrect keys in `__all__`.'


def test_registry():
    """Check smoothness regularizations are added to LOSS_REGISTRY."""
    assert 'disp_smooth' in LOSS_REG, "Missing key from loss registry."
    assert LOSS_REG['disp_smooth'] == SmoothReg, "Incorrect class in loss registry."

    assert 'feat_peaky' in LOSS_REG, "Missing key from loss registry."
    assert LOSS_REG['feat_peaky'] == FeatPeakReg, "Incorrect class in loss registry."

    assert 'feat_smooth' in LOSS_REG, "Missing key from loss registry."
    assert LOSS_REG['feat_smooth'] == FeatSmoothReg, "Incorrect class in loss registry."
