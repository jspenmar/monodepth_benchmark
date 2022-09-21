import numpy as np
import torch

from src.tools import viz
from src.tools.viz import *


def test_all():
    """Check all expected symbols are imported."""
    items = {'rgb_from_disp', 'rgb_from_feat'}
    assert set(viz.__all__) == items, 'Incorrect keys in `__all__`.'


# -----------------------------------------------------------------------------
class TestRGBfromDisp:
    def test_default(self):
        x = torch.rand(2, 1, 10, 20)

        out = rgb_from_disp(x)
        out2 = rgb_from_disp(x, cmap='turbo', vmin=0, vmax=[np.percentile(x[0], 95), np.percentile(x[1], 95)])

        assert np.allclose(out, out2), "Incorrect default params."

    def test_range(self):
        """Test disparity conversion with custom normalization ranges."""
        arr = np.array([[0, 0, 0.5, 0.5, 1, 1]])
        out = rgb_from_disp(arr).squeeze()

        out2 = rgb_from_disp(arr, vmin=0.5, vmax=1).squeeze()
        assert np.allclose(out2[2], out2[3]), 'Incorrect sanity check for same value.'
        assert not np.allclose(out2[3], out2[4]), 'Incorrect sanity check for different value.'
        assert np.allclose(out2[0], out2[2]), 'Incorrect clipping to min value.'
        assert np.allclose(out[0], out2[0]), 'Inconsistent min value.'
        assert not np.allclose(out2[2], out[2]), 'Incorrect clipping to min value.'

        out3 = rgb_from_disp(arr, vmin=0, vmax=0.5).squeeze()
        assert np.allclose(out3[2], out3[3]), 'Incorrect sanity check for same value.'
        assert not np.allclose(out3[2], out3[0]), 'Incorrect sanity check for different value.'
        assert np.allclose(out3[2], out3[4]), 'Incorrect clipping to max value.'
        assert np.allclose(out[5], out3[5]), 'Inconsistent max value.'
        assert not np.allclose(out3[2], out[2]), 'Incorrect clipping to max value.'

    def test_inv(self):
        x = torch.rand(2, 1, 10, 20)
        x_inv = 1/x

        out = rgb_from_disp(x, invert=True)
        out2 = rgb_from_disp(x_inv, invert=False)
        assert np.allclose(out, out2), "Incorrect inversion."

    def test_shape(self):
        x = torch.rand(1, 1, 10, 20)

        out = rgb_from_disp(x)
        out2 = rgb_from_disp(x.squeeze())
        assert np.allclose(out[0], out2), "Incorrect out with different ndim."
        assert out.ndim == 4, "Incorrect dim for 4D input."
        assert out2.ndim == 3, "Incorrect dim for 2D input."

    def test_np(self):
        x = torch.rand(2, 1, 10, 20)
        x_np = x.permute(0, 2, 3, 1).numpy()

        out = rgb_from_disp(x)
        out = out.permute(0, 2, 3, 1).numpy()

        out2 = rgb_from_disp(x_np)
        assert np.allclose(out, out2), "Incorrect conversion to np."


# -----------------------------------------------------------------------------
class TestRGBfromFeat:
    def test_norm(self):
        x = torch.rand(1, 5, 10, 20)

        out = rgb_from_feat(x).squeeze().flatten(-2, -1)
        assert torch.allclose(out.min(-1)[0], out.new_zeros(3)), "Incorrect min norm."
        assert torch.allclose(out.max(-1)[0], out.new_ones(3)), "Incorrect max norm."

    def test_shape(self):
        x = torch.rand(1, 5, 10, 20)

        out = rgb_from_feat(x)
        assert out.shape[1] == 3, "Expected output to be RGB."

        out2 = rgb_from_feat(x[0])
        assert out2.shape[0] == 3, "Expected output to be RGB."
        assert torch.allclose(out[0], out2), "Incorrect output with different dimensions.."

    def test_np(self):
        x = torch.rand(2, 5, 10, 20)
        x_np = x.permute(0, 2, 3, 1).numpy()

        out = rgb_from_feat(x)
        out = out.permute(0, 2, 3, 1).numpy()

        out2 = rgb_from_feat(x_np)
        assert np.allclose(out, out2), "Incorrect conversion to np."

