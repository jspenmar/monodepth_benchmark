"""
Tests for `src/registry.py`.
"""
import pytest

from src import registry
from src.registry import *


def test_all():
    """Check all expected symbols are imported."""
    items = {'register', 'NET_REG', 'DATA_REG', 'LOSS_REG', 'SCHED_REG'}
    assert set(registry.__all__) == items, 'Incorrect keys in `__all__`.'


def test_sched():
    """Check scheduler registry has all expected keys."""
    keys = {'steplr', 'exp', 'cos', 'cos_warm', 'plateau'}
    assert set(SCHED_REG.keys()) == keys, f'Incorrect SCHEDULER keys.'


def test_add_net():
    """Check adding to the network registry."""
    name, type = 'test', 'net'

    @register(name, type)
    class Test: ...
    assert name in NET_REG, 'Missing item from NETWORK registry.'
    NET_REG.pop(name)


def test_add_loss():
    """Check adding to the loss registry."""
    name, type = 'test', 'loss'

    @register(name, type)
    class Test: ...
    assert name in LOSS_REG, 'Missing item from LOSS registry.'
    LOSS_REG.pop(name)


def test_add_dataset():
    """Check adding to the dataset registry."""
    name, type = 'test', 'data'

    @register(name, type)
    class Test: ...
    assert name in DATA_REG, 'Missing item from DATASET registry.'
    DATA_REG.pop(name)


def test_add_auto():
    """Check automatic adding based on class name."""
    name = 'test'

    @register(name)
    class TestNet: ...
    assert name in NET_REG, 'Missing item from automatic NET registry.'
    NET_REG.pop(name)

    @register(name)
    class TestLoss: ...
    assert name in LOSS_REG, 'Missing item from automatic LOSS registry.'
    LOSS_REG.pop(name)

    @register(name+'2')
    class TestReg: ...
    assert name +'2' in LOSS_REG, 'Missing item from automatic LOSS registry.'
    LOSS_REG.pop(name + '2')

    @register(name)
    class TestDataset: ...
    assert name in DATA_REG, 'Missing item from automatic DATA registry.'
    DATA_REG.pop(name)

    with pytest.raises(ValueError):
        @register(name)
        class Test: ...


def test_add_multiple():
    name = ('test1', 'test2')

    @register(name, type='net')
    class TestNet: ...

    for n in name: assert n in NET_REG, 'Missing item from automatic NET registry.'
    [NET_REG.pop(n) for n in name]


def test_register_types():
    """Check raised exception when adding to unknown registry."""
    with pytest.raises(TypeError):
        @register(name='temp', type='foo')
        class Test: ...


def test_register_duplicates():
    """Check raised exception when registering a duplicate name."""
    name, type = 'test', 'net'
    @register(name, type)
    class Test: ...

    # Should not overwrite by default.
    with pytest.raises(ValueError):
        @register(name, type)
        class Test2: ...

    # Explicit check.
    with pytest.raises(ValueError):
        @register(name, type, overwrite=False)
        class Test3: ...

    # Explicit check with multiple.
    with pytest.raises(ValueError):
        @register(('asdf', name), type, overwrite=False)
        class Test3: ...

    # Explicit check with multiple.
    with pytest.raises(ValueError):
        @register((name, 'asdf'), type, overwrite=False)
        class Test3: ...

    NET_REG.pop(name)


def test_register_overwrite():
    """Check registry items can be overwritten if desired."""
    name, type = 'test', 'net'

    @register(name, type)
    class Test: ...
    assert NET_REG[name] == Test, 'Unexpected base class when overwriting.'

    @register(name, type, overwrite=True)
    class Test2: ...
    assert NET_REG[name] == Test2, 'Unexpected overwritten class.'

    NET_REG.pop(name)


def test_ignore_main():
    """Check classes created in `__main__` are ignored."""
    from unittest.mock import Mock
    name, type = 'test', 'loss'

    Mock.__module__ = '__main__'
    with pytest.warns(UserWarning):
        _ = register(name, type)(Mock)

    assert name not in LOSS_REG, 'Class from `__main__` not ignored.'

