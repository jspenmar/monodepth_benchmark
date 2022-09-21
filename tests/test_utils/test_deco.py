import inspect

import pytest

from src.utils.deco import *
from src.utils import deco, MultiLevelTimer


def test_all():
    """Check all expected symbols are imported."""
    items = {'opt_args_deco', 'delegates', 'map_container', 'retry_new_on_error'}
    assert set(deco.__all__) == items, 'Incorrect keys in `__all__`.'


# -----------------------------------------------------------------------------
@opt_args_deco
def _deco(func, prefix='', suffix=''):
    """Helper."""
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        return out, f'{prefix}{out}{suffix}'
    return wrapper


def _add(a, b):
    """Helper."""
    return a + b


class TestOptArgsDeco:
    def test_base(self):
        """Test different ways of instantiating optional arguments."""
        func = _deco(_add)
        assert func(1, 2) == (3, '3'), "Error with default arguments."

        func = _deco(_add, prefix='***')
        assert func(1, 2) == (3, '***3'), "Error with first default arg."

        func = _deco(prefix='***', suffix='***')(_add)
        assert func(1, 2) == (3, '***3***'), "Error with default args."

    def test_callable(self):
        """Test that we raise errors to enforce keyword-only optional arguments."""
        with pytest.raises(TypeError):
            _ = _deco(_add, '***')  # Optional arguments should be keyword-only

        with pytest.raises(TypeError):
            _ = _deco('***')(_add)  # Check that positional arg is callable


# -----------------------------------------------------------------------------
def _parent(a, b=0, c=None, **kwargs): ...
def _child(new, *args, **kwargs): ...


class TestDelegates:
    def test_default(self):
        fn = delegates(_parent)(_child)

        sig = inspect.signature(fn)
        sigd = dict(sig.parameters)
        assert list(sigd) == ['new', 'a', 'b', 'c'], 'Incorrect delegated signature.'


# -----------------------------------------------------------------------------
def _stringify(x, suffix=''):
    """Helper."""
    return f'{x}{suffix}'


class TestMapContainer:
    def test_single(self):
        """Test with a single input, i.e. equivalent to the original function."""
        input = 2
        func = map_container(_stringify)
        assert func(input) == _stringify(input), "Error in 'map_apply' single input"
        assert func(input, suffix='***') == _stringify(input, suffix='***'), "Error in 'map_apply' single input"

    def test_multi(self):
        """Test using nested sequences/dicts."""
        input = [2, {
            'a': 'test',
            'b': {1},
            'c': (1, 2),
        }]

        target1 = ['2', {
            'a': 'test',
            'b': {'1'},
            'c': ('1', '2'),
        }]

        target2 = ['2***', {
            'a': 'test***',
            'b': {'1***'},
            'c': ('1***', '2***'),
        }]

        func = map_container(_stringify)
        assert func(input) == target1, "Error in 'map_apply' sequence input"
        assert func(input, suffix='***') == target2, "Error in 'map_apply' sequence input"


# -----------------------------------------------------------------------------
class Dataset:
    def __init__(self, n):
        self.n = n
        self.log_time = True
        self.timer = MultiLevelTimer()

        self.__class__.__getitem__ = retry_new_on_error(
            self.__class__.getitem,
            exc=self.retry_exc,
            silent=self.silent,
            max=self.max,
            use_blacklist=self.use_blacklist
        )

    def __init_subclass__(cls, retry_exc, silent, max_retries, use_blacklist, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.retry_exc = retry_exc
        cls.silent = silent
        cls.max = max_retries
        cls.use_blacklist = use_blacklist

    def __len__(self): return self.n
    def getitem(self, item): raise NotImplementedError
    def __getitem__(self, item): return self.getitem(item)


class TestRetryDifferentOnError:
    def test_default(self):
        """Test default parameters, catching any exception and logging."""
        class TmpData(Dataset, retry_exc=Exception, silent=False, max_retries=None, use_blacklist=False):
            def getitem(self, item):
                if item % 2 == 0: raise ValueError
                return {'item': item}, {}, {'item': str(item)}

        dataset = TmpData(10)

        x, y, meta = dataset[1]
        assert x['item'] == 1, "Loading of item (without exception) failed."
        assert 'errors' in meta, "Missing 'error' key when logging errors"

        x, y, meta = dataset[2]
        assert x['item'] != 2, "Loading of item (with exception) failed."

    def test_exc_single(self):
        """Test that we can catch a specific exception."""
        class TmpData(Dataset, retry_exc=ValueError, silent=False, max_retries=None, use_blacklist=False):
            def getitem(self, item):
                if item % 2 == 0: raise ValueError
                return {'item': item}, {}, {'item': str(item)}

        x, y, meta = TmpData(10)[2]
        assert x['item'] != 2, "Loading of item (with exception) failed."

    def test_exc_ignore(self):
        """Test that we ignore non-specified exceptions."""
        class TmpData(Dataset, retry_exc=ValueError, silent=False, max_retries=None, use_blacklist=False):
            def getitem(self, item):
                if item % 2 == 0: raise TypeError
                return {'item': item}, {}, {'item': str(item)}

        dataset = TmpData(10)
        _ = dataset[1]

        with pytest.raises(TypeError):
            _ = dataset[2]

    def test_exc_multiple(self):
        """Test that we can catch multiple specific exceptions."""
        class TmpData(Dataset, retry_exc=[ValueError, TypeError], silent=False, max_retries=None, use_blacklist=False):
            def getitem(self, item):
                if item % 2 == 0: raise ValueError
                if item % 3 == 0: raise TypeError
                return {'item': item}, {}, {'item': str(item)}

        dataset = TmpData(10)
        x, y, meta = dataset[2]
        assert x['item'] != 2, "Loading of item (with exception) failed."

        x, y, meta = dataset[3]
        assert x['item'] != 2, "Loading of item (with exception) failed."

    def test_exc_none(self):
        """Test that we can disable all exceptions."""
        class TmpData(Dataset, retry_exc=None, silent=False, max_retries=None, use_blacklist=False):
            def getitem(self, item):
                if item % 2 == 0: raise ValueError
                if item % 3 == 0: raise TypeError
                return {'item': item}, {}, {'item': str(item)}

        dataset = TmpData(10)
        _ = dataset[5]

        with pytest.raises(ValueError):
            _ = dataset[2]

        with pytest.raises(TypeError):
            _ = dataset[3]

    def test_silent(self):
        class TmpData(Dataset, retry_exc=Exception, silent=True, max_retries=None, use_blacklist=False):
            def getitem(self, item):
                if item % 2 == 0: raise ValueError
                return {'item': item}, {}, {'item': str(item)}

        dataset = TmpData(10)
        x, y, meta = dataset[2]
        assert x['item'] != 2,  "Loading of item (with exception) failed."
        assert 'errors' not in meta, "Error when disabling exception catching."

    def test_max_retries(self):
        class TmpData(Dataset, retry_exc=Exception, silent=False, max_retries=None, use_blacklist=False):
            def getitem(self, item):
                raise ValueError

        with pytest.raises(RecursionError):
            _ = TmpData(10)[0]

        class TmpData(Dataset, retry_exc=Exception, silent=False, max_retries=5, use_blacklist=False):
            def getitem(self, item):
                raise ValueError

        with pytest.raises(RuntimeError):
            _ = TmpData(10)[0]

    def test_blacklist(self):
        """Test that we can add items to a blacklist, which are excluded from reloading."""
        class TmpData(Dataset, retry_exc=Exception, silent=False, max_retries=None, use_blacklist=True):
            def getitem(self, item):
                if item != 0:
                    raise ValueError
                return {'item': item}, {}, {'item': str(item)}

        dataset = TmpData(10)
        _ = [dataset[i] for i in range(10)]  # Add all items to the blacklist

        for i in range(10):
            x, y, meta = dataset[i]
            assert x['item'] == 0
            for j in range(10):
                if i != j:
                    assert str(j) not in meta['errors'], "Error including item in blacklist."

    def test_blacklist_none(self):
        """Test that items causing exceptions can be repeated when not creating a blacklist."""

        class TmpData(Dataset, retry_exc=Exception, silent=False, max_retries=None, use_blacklist=False):
            def getitem(self, item):
                if item != 0:
                    raise ValueError
                return {'item': item}, {}, {'item': str(item)}

        dataset = TmpData(10)
        num_errors = []
        for i in range(10):
            x, y, meta = dataset[i]
            assert x['item'] == 0, "Error loading correct item."
            num_errors.append(meta['errors'].count('ValueError'))
        assert max(num_errors) > 1, "Error when repeating exception items."
