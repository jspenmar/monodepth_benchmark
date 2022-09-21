import numpy as np
import pytest
import torch

from src.tools import geometry
from src.tools.geometry import *


def test_all():
    """Check all expected symbols are imported."""
    items = {
        'extract_edges', 'to_scaled', 'to_log', 'to_inv', 'blend_stereo',
        'T_from_Rt', 'T_from_AAt',
        'BackprojectDepth', 'ProjectPoints', 'ViewSynth',
    }
    assert set(geometry.__all__) == items, 'Incorrect keys in `__all__`.'


# -----------------------------------------------------------------------------
class TestExtractEdges:
    pass


# -----------------------------------------------------------------------------
class TestBlendStereo:
    def test_accuracy(self):
        pass

    def test_shape(self):
        l, r = torch.rand(1, 1, 10, 20), torch.rand(1, 1, 10, 20)

        out1 = blend_stereo(l, r)
        out2 = blend_stereo(l[0], r[0])
        out3 = blend_stereo(l[0, 0], r[0, 0])

        assert torch.allclose(out1[0, 0], out2[0]), "Error when computing with 4D input."
        assert torch.allclose(out2[0], out3), "Error when computing with 3D input."

    def test_np(self):
        l, r = torch.rand(1, 1, 10, 20), torch.rand(1, 1, 10, 20)
        l_np, r_np = l.permute(0, 2, 3, 1).numpy(), r.permute(0, 2, 3, 1).numpy()

        out1 = blend_stereo(l, r)
        out1 = out1.permute(0, 2, 3, 1).numpy()

        out2 = blend_stereo(l_np, r_np)
        assert np.allclose(out1, out2), "Incorrect blend stereo to numpy with permute."


# -----------------------------------------------------------------------------
class TestToScaledDepth:
    def test_accuracy(self):
        x = torch.rand(2, 1, 10, 20)
        min, max = 0.01, 100
        i_max, i_min = 1/min, 1/max

        tgt_min = (i_max-i_min)*x.min() + i_min
        tgt_max = (i_max-i_min)*x.max() + i_min

        out11, out12 = to_scaled(x, min=0.01, max=100)

        assert out11.max() == tgt_max, 'Incorrect max scaled disp.'
        assert out11.min() == tgt_min, 'Incorrect min scaled disp.'

        assert out12.max() == 1/tgt_min, 'Incorrect max scaled depth.'
        assert out12.min() == 1/tgt_max, 'Incorrect min scaled depth.'

    def test_default(self):
        x = torch.rand(2, 1, 10, 20)

        out11, out12 = to_scaled(x)
        out21, out22 = to_scaled(x, min=0.01, max=100)
        assert torch.allclose(out11, out21), 'Incorrect default parameters'
        assert torch.allclose(out12, out22), 'Incorrect default parameters'

    def test_shape(self):
        """Test arbitrary input size and consistent results."""
        b, n = 4, 2
        x = torch.rand(b, n, 1, 10, 20) + 0.2

        out1 = to_scaled(x)[0]  # 5D output
        out2 = to_scaled(x[0])[0]  # 4D output
        out3 = to_scaled(x[0, 0])[0]  # 3D output
        out4 = to_scaled(x[0, 0, 0])[0]  # 2D output

        assert torch.allclose(out1[0, 0, 0], out2[0, 0]), "Error when computing with 5D input."
        assert torch.allclose(out2[0, 0], out3[0]), "Error when computing with 4D input."
        assert torch.allclose(out3[0], out4), "Error when computing with 3D input."

    def test_np(self):
        x = torch.rand(2, 1, 10, 20)
        x_np = x.permute(0, 2, 3, 1).numpy()

        out11, out12 = to_scaled(x)
        out11 = out11.permute(0, 2, 3, 1).numpy()
        out12 = out12.permute(0, 2, 3, 1).numpy()

        out21, out22 = to_scaled(x_np)
        assert np.allclose(out11, out21), "Incorrect scaled disp to numpy with permute."
        assert np.allclose(out12, out22), "Incorrect scaled depth to numpy with permute."


# -----------------------------------------------------------------------------
class TestToLogDepth:
    def test_accuracy(self):
        x = torch.rand(2, 1, 10, 20) + 0.2  # Ensure no values are close to zero.
        tgt = x.log()

        out = to_log(x)
        assert torch.allclose(out, tgt), "Incorrect conversion to log depth."

    def test_mask(self):
        x = torch.rand(2, 1, 10, 20)
        tgt = x.log()

        mask = x < 0.5
        x, tgt = x*mask, tgt*mask

        out = to_log(x)
        assert torch.allclose(out, tgt), "Incorrect masked conversion to log depth."

    def test_shape(self):
        """Test arbitrary input size and consistent results."""
        b, n = 4, 2
        x = torch.rand(b, n, 1, 10, 20) + 0.2

        out1 = to_log(x)  # 5D output
        out2 = to_log(x[0])  # 4D output
        out3 = to_log(x[0, 0])  # 3D output
        out4 = to_log(x[0, 0, 0])  # 2D output

        assert torch.allclose(out1[0, 0, 0], out2[0, 0]), "Error when computing with 5D input."
        assert torch.allclose(out2[0, 0], out3[0]), "Error when computing with 4D input."
        assert torch.allclose(out3[0], out4), "Error when computing with 3D input."

    def test_np(self):
        x = torch.rand(2, 1, 10, 20)
        x_np = x.permute(0, 2, 3, 1).numpy()

        out1 = to_log(x)
        out1 = out1.permute(0, 2, 3, 1).numpy()

        out2 = to_log(x_np)
        assert np.allclose(out1, out2), "Incorrect conversion to numpy with permute."


# -----------------------------------------------------------------------------
class TestToDisp:
    def test_accuracy(self):
        x = torch.rand(2, 1, 10, 20) + 0.2  # Ensure no values are close to zero.
        tgt = 1/x

        out = to_inv(x)
        assert torch.allclose(out, tgt), "Incorrect conversion to disparity."

        out2 = to_inv(out)
        assert torch.allclose(out2, x), "Incorrect disparity cycle consistency."

    def test_mask(self):
        x = torch.rand(2, 1, 10, 20)
        tgt = 1/x

        mask = x < 0.5
        x, tgt = x*mask, tgt*mask

        out = to_inv(x)
        assert torch.allclose(out, tgt), "Incorrect masked conversion to disparity."

        out2 = to_inv(out)
        assert torch.allclose(out2, x), "Incorrect masked disparity cycle consistency."

    def test_shape(self):
        """Test arbitrary input size and consistent results."""
        b, n = 4, 2
        x = torch.rand(b, n, 1, 10, 20) + 0.2

        out1 = to_inv(x)  # 5D output
        out2 = to_inv(x[0])  # 4D output
        out3 = to_inv(x[0, 0])  # 3D output
        out4 = to_inv(x[0, 0, 0])  # 2D output

        assert torch.allclose(out1[0, 0, 0], out2[0, 0]), "Error when computing with 5D input."
        assert torch.allclose(out2[0, 0], out3[0]), "Error when computing with 4D input."
        assert torch.allclose(out3[0], out4), "Error when computing with 3D input."

    def test_np(self):
        x = torch.rand(2, 1, 10, 20)
        x_np = x.permute(0, 2, 3, 1).numpy()

        out1 = to_inv(x)
        out1 = out1.permute(0, 2, 3, 1).numpy()

        out2 = to_inv(x_np)
        assert np.allclose(out1, out2), "Incorrect conversion to numpy with permute."


# -----------------------------------------------------------------------------
class TestTfromAAt:
    def test_accuracy(self):
        """Test accurate conversion. Example from https://ch.mathworks.com/help/robotics/ref/axang2rotm.html"""
        aa = torch.tensor([[0, torch.pi/2, 0]])
        t = torch.rand(1, 3)
        tgt = aa.new_tensor([[
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
        ]])

        out = T_from_AAt(aa, t)
        assert torch.allclose(out[:, :3, 3], t), "Incorrect translation vector."
        assert torch.allclose(out[:, :3, :3], tgt), "Incorrect rotation matrix."

    def test_shape(self):
        """Test arbitrary input size and consistent results."""
        b, n = 4, 2
        aa, t = torch.rand(b, n, 3), torch.rand(b, n, 3)

        out1 = T_from_AAt(aa, t)  # 5D output
        out2 = T_from_AAt(aa[0], t[0])  # 4D output
        out3 = T_from_AAt(aa[0, 0], t[0, 0])  # 3D output

        assert torch.allclose(out1[0, 0], out2[0]), "Error when computing with 4D input."
        assert torch.allclose(out2[0], out3), "Error when computing with 2D input."

    def test_shape_raise(self):
        """Test exceptions are raise when using incorrect or non-mathing shapes."""
        b, n = 4, 2
        aa, t = torch.rand(b, n, 3), torch.rand(b, n, 3)

        with pytest.raises(ValueError):
            T_from_AAt(torch.rand(5), t)

        with pytest.raises(ValueError):
            T_from_AAt(aa, torch.rand(5))

        with pytest.raises(ValueError):
            T_from_AAt(torch.rand(2, 3), torch.rand(4, 3))

    def test_np(self):
        """Test numpy output is consistent with torch and is not permuted."""
        aa = torch.rand(4, 3)
        t = torch.rand(4, 3)

        out1 = T_from_AAt(aa, t)  # 5D output
        out2 = T_from_AAt(aa.numpy(), t.numpy())

        assert np.allclose(out1.numpy(), out2), "Error with numpy input."


# -----------------------------------------------------------------------------
class TestTfromRt:
    def test_accuracy(self):
        R = torch.rand(3, 3)
        t = torch.rand(3)

        out = T_from_Rt(R, t)
        assert torch.allclose(out[:3, :3], R), "Incorrect rotation matrix."
        assert torch.allclose(out[:3, 3], t), "Incorrect translation vector."

    def test_shape(self):
        b, n = 4, 2
        R = torch.rand(b, n, 3, 3)
        t = torch.rand(b, n, 3)

        out1 = T_from_Rt(R, t)
        out2 = T_from_Rt(R[0], t[0])
        out3 = T_from_Rt(R[0, 0], t[0, 0])

        assert torch.allclose(out1[0, 0], out2[0]), "Error when computing with 4D input."
        assert torch.allclose(out2[0], out3), "Error when computing with 2D input."

    def test_shape_raises(self):
        b, n = 4, 2
        R = torch.rand(b, n, 3, 3)
        t = torch.rand(b, n, 3)

        with pytest.raises(ValueError):
            T_from_Rt(torch.rand(5), t)

        with pytest.raises(ValueError):
            T_from_Rt(R, torch.rand(5))

        with pytest.raises(ValueError):
            T_from_Rt(torch.rand(2, 3, 3), torch.rand(4, 3))

    def test_np(self):
        """Test numpy output is consistent with torch and is not permuted."""
        R = torch.rand(4, 3, 3)
        t = torch.rand(4, 3)

        out1 = T_from_Rt(R, t)
        out2 = T_from_Rt(R.numpy(), t.numpy())

        assert np.allclose(out1.numpy(), out2), "Error with numpy input."
