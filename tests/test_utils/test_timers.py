import time
from unittest import mock

import numpy as np
import pytest

from src.utils.timers import *
from src.utils import timers


def test_all():
    """Check all expected symbols are imported."""
    items = {'Timer', 'MultiLevelTimer'}
    assert set(timers.__all__) == items, 'Incorrect keys in `__all__`.'


# -----------------------------------------------------------------------------
class TestTimer:
    def test_options(self):
        """Test that formatting options are set correctly."""
        name, precision = 'Test', 4
        timer = Timer(name=name, as_ms=True, precision=precision)
        assert timer.name == name, "Incorrect Timer name"
        assert repr(timer) == f'Timer(name={name}, as_ms=True, sync_gpu=False, precision={precision})'

        with timer as t: ...
        parts = str(t).split(' ')
        assert len(parts) == 3, "Incorrect Timer formatting, expected 'name: elapsed units'"
        assert parts[2] == 'ms', "Incorrect Timer units."
        assert len(parts[1].split('.')[-1]) <= precision, "Incorrect Timer precision."

    def test_accuracy(self):
        """Test that timing is accurate (within 2ms)."""
        target = 0.3
        with Timer() as t:
            time.sleep(target)

        assert np.allclose(t.elapsed, target, atol=0.02), "Timer off by more than 2ms."

    @mock.patch('torch.cuda.synchronize')
    def test_sync_gpu(self, sync_fn):
        """Test that torch synchronize is called correctly."""
        with Timer(sync_gpu=True): ...
        assert sync_fn.call_count == 2, "Incorrect number of calls to synchronize."


# -----------------------------------------------------------------------------
class TestMultiLevelTimer:
    def test_options(self):
        """Test that formatting options are set correctly."""
        name, precision = 'Test', 4
        timer = MultiLevelTimer(name=name, as_ms=True, sync_gpu=False, precision=precision)
        assert repr(timer) == f'MultiLevelTimer(name={name}, as_ms=True, sync_gpu=False, precision={precision})'
        _ = str(timer)  # No good way of testing, since it varies depending on time taken

    def test_keys(self):
        """Test that timer keys are set correctly."""
        timer = MultiLevelTimer()
        with timer('Label'):
            assert 'Label' in timer._data, "Error setting label name."
            with pytest.raises(KeyError):
                _ = timer('Label').__enter__()  # Fail with duplicate keys

        timer = MultiLevelTimer()
        with timer: ...
        assert 'Level1' in timer._data, "Error setting default label name."

    def test_accuracy(self):
        """Test accuracy with single label."""
        target = 0.3

        timer = MultiLevelTimer()
        with timer('Test'):
            time.sleep(target)

        assert np.allclose(timer['Test']['elapsed'], target, atol=0.02), "Timer off by more than 2ms."

    def test_nesting(self):
        """Test accuracy with nesting."""
        target = 0.3

        timer = MultiLevelTimer()
        with timer('Test1'):
            time.sleep(target)
            with timer('Test2'):
                time.sleep(target*2)

        assert timer['Test2']['depth'] == 2, "Incorrect 'inner' depth level."
        assert timer['Test1']['depth'] == 1, "Incorrect 'outer' depth level."

        target2 = target*2
        target1 = target + target2
        assert np.allclose(timer['Test2']['elapsed'], target2, atol=0.02), "'Inner' off by more than 2ms."
        assert np.allclose(timer['Test1']['elapsed'], target1, atol=0.02), "'Outer' off by more than 2ms."

    @mock.patch('torch.cuda.synchronize')
    def test_sync_gpu(self, sync_fn):
        """Test that torch synchronize is called correctly."""
        timer = MultiLevelTimer(sync_gpu=True)
        with timer: ...
        assert sync_fn.call_count == 2, "Incorrect number of calls to synchronize."

    def test_copy(self):
        """Test that timing data is copied correctly and is independent of original timer."""
        timer = MultiLevelTimer(name='Test', as_ms=False, precision=4)
        with timer:
            time.sleep(0.1)

        data = timer.copy()._data
        assert data == timer._data, "Incorrect data copied."
        assert data is not timer._data, "Incorrect deep copy of data."

        data['test'] = 0
        assert 'test' not in timer._data, "Incorrect deep copy of data."

    def test_reset(self):
        """Test that timer data can be reset."""
        timer = MultiLevelTimer()
        with timer('Label'): ...
        assert 'Label' in timer._data
        timer.reset()
        assert timer._data == {}, "Incorrect data deletion."

        with timer('Label'):
            with pytest.raises(RuntimeError):
                timer.reset()  # Error resetting while running timer.

    def test_mean_elapsed(self):
        """Test that timers elapsed time get averaged correctly."""
        assert MultiLevelTimer.mean_elapsed([]) == [], "Error returning empty list."
        assert MultiLevelTimer.mean_elapsed(None) is None, "Error returning None."

        sleep_time1, sleep_time2 = 0.3, 0.9
        target = (sleep_time1+sleep_time2) / 2

        timer1 = MultiLevelTimer()
        with timer1('Test'):
            time.sleep(sleep_time1)

        timer2 = MultiLevelTimer()
        with timer2('Test'):
            time.sleep(sleep_time2)

        data = MultiLevelTimer.mean_elapsed([timer1, timer2])
        assert np.allclose(data['Test'], target, atol=0.02), "'mean_elapsed' off by more than 2ms."
