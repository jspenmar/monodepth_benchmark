import logging

import numpy as np
import pytest

from src.utils.misc import *
from src.utils import misc


def test_all():
    """Check all expected symbols are imported."""
    items = {'get_logger', 'flatten_dict', 'sort_dict', 'apply_cmap'}
    assert set(misc.__all__) == items, 'Incorrect keys in `__all__`.'


# -----------------------------------------------------------------------------
class TensorGetLogger:
    def test_default(self):
        key = 'test1234'
        logger = get_logger(key)

        assert key in logging.root.manager.loggerDict, 'Logger not created.'
        assert logger == logging.root.manager.loggerDict[key], 'Incorrect logger created.'

        assert not logger.propagate, 'Logger propagate should be disabled.'
        assert len(logger.handlers) == 1, 'Incorrect number of handlers.'

    def test_duplicate(self):
        """Test that we do not duplicate logger StreamHandlers."""
        # Test that default behaviour does add two handlers
        name = 'test_duplicate'
        logger = logging.getLogger(name)
        logger.addHandler(logging.StreamHandler())
        assert len(logger.handlers) == 1, "Logger should only have one handler."

        logger.addHandler(logging.StreamHandler())
        assert len(logger.handlers) == 2, "Logger should have two handlers."

        # Test that our function only adds one
        name2 = 'test_duplicate_v2'
        logger2 = get_logger(name2)
        assert len(logger2.handlers) == 1, "Custom Logger should only have one handler."

        logger2 = get_logger(name2)
        assert len(logger2.handlers) == 1, "Custom Logger should only have one handler (after second time)."


# -----------------------------------------------------------------------------
class TestFlatten:
    def test_default(self):
        """Test basic nesting & default separator."""
        d = {'a': 1, 'b': 2, 'c': dict(a=1, b=2, )}
        tgt = {'a': 1, 'b': 2, 'c/a': 1, 'c/b': 2}

        out = flatten_dict(d)
        assert out == tgt, "Incorrect flattened keys."

        out = flatten_dict(d, sep='/')
        assert out == tgt, "Incorrect separator keys."

    def test_separator(self):
        """Test custom separator."""
        d = {'a': 1, 'b': 2, 'c': dict(a=1, b=2, )}
        tgt = {'a': 1, 'b': 2, 'c.a': 1, 'c.b': 2}

        out = flatten_dict(d, sep='.')
        assert out == tgt, "Incorrect flattened keys."

    def test_nesting(self):
        """Test multiple nestings."""
        d = {
            'a': [0, 1, {}],
            'b': {'a': 1, 'b': 2, 'c': []},
            'c': {'a': {'a': 0}, 'b': {'b': 1}, 'c': {'c': 2}}
        }
        tgt = {
            'a': [0, 1, {}],
            'b/a': 1, 'b/b': 2, 'b/c': [],
            'c/a/a': 0, 'c/b/b': 1, 'c/c/c': 2,
        }

        out = flatten_dict(d)
        assert out == tgt, "Incorrect flattened keys."


# -----------------------------------------------------------------------------
class TestSortedDict:
    def test_sorted_dict(self):
        """Test sorting dict keys."""
        d = dict(b=2, c=1, a=10)
        tgt = dict(a=10, b=2, c=1)

        out = sort_dict(d)
        assert out == tgt, 'Incorrect sorted order'

        with pytest.raises(TypeError):
            sort_dict({'a': 1, 1: 0, 'test': None})


# -----------------------------------------------------------------------------
class TestApplyCmap:
    def test_default(self):
        """Test applying a cmap with default parameters."""
        arr = np.array([0, 0, 0.5, 0.5, 1, 1])
        out = apply_cmap(arr)

        assert np.allclose(out[0], out[1]), 'Incorrect 0 mapping.'
        assert np.allclose(out[2], out[3]), 'Incorrect 0.5 mapping.'
        assert np.allclose(out[4], out[5]), 'Incorrect 1 mapping.'
        assert not np.allclose(out[0], out[2]), 'Incorrect 0 vs. 0.5'
        assert not np.allclose(out[0], out[4]), 'Incorrect 0 vs. 1'
        assert not np.allclose(out[2], out[4]), 'Incorrect 0.5 vs. 1'

        out2 = apply_cmap(arr, cmap='turbo')
        assert np.allclose(out, out2), 'Incorrect default colormap, expected "turbo".'

        out3 = apply_cmap(arr, vmin=arr.min(), vmax=arr.max())
        assert np.allclose(out, out3), 'Incorrect default range.'

    def test_range(self):
        """Test applying a cmap with custom normalization ranges."""
        arr = np.array([0, 0, 0.5, 0.5, 1, 1])
        out = apply_cmap(arr)

        out2 = apply_cmap(arr, vmin=0.5)
        assert np.allclose(out2[2], out2[3]), 'Incorrect sanity check for same value.'
        assert not np.allclose(out2[3], out2[4]), 'Incorrect sanity check for different value.'
        assert np.allclose(out2[0], out2[2]), 'Incorrect clipping to min value.'
        assert np.allclose(out[0], out2[0]), 'Inconsistent min value.'
        assert not np.allclose(out2[2], out[2]), 'Incorrect clipping to min value.'

        out3 = apply_cmap(arr, vmax=0.5)
        assert np.allclose(out3[2], out3[3]), 'Incorrect sanity check for same value.'
        assert not np.allclose(out3[2], out3[0]), 'Incorrect sanity check for different value.'
        assert np.allclose(out3[2], out3[4]), 'Incorrect clipping to max value.'
        assert np.allclose(out[5], out3[5]), 'Inconsistent max value.'
        assert not np.allclose(out3[2], out[2]), 'Incorrect clipping to max value.'
