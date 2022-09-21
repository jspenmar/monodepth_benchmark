import copy
import inspect
import time
from typing import Callable, Generator, Optional, Sequence, Union

from src.typing import TimerData

__all__ = ['Timer', 'MultiLevelTimer']


class Timer:
    """Context manager for timing a block of code.

    Attributes:
    :param name: (str) Timer label when printing.
    :param as_ms: (bool) If `True`, store time as `milliseconds`, otherwise `seconds`.
    :param sync_gpu: (bool) If `True`, ensure that GPU is synced on Timer enter and exit.
    :param precision: (int) Number of decimal places to print.

    Example:
    ```
        with Timer('MyTimer') as t:
            time.sleep(1)
        elapsed = t.elapsed
        print(t)

        ===>
        MyTimer: 1.003 s
    ```
    """
    def __init__(self, name: str = 'Timer', as_ms: bool = False, sync_gpu: bool = False, precision: int = 6) -> None:
        self.name: str = name
        self.as_ms: bool = as_ms
        self.sync_gpu: bool = sync_gpu
        self.precision: int = precision

        self._sf: int = 1000 if self.as_ms else 1
        self._units: str = 'ms' if self.as_ms else 's'
        self._sync_fn: Optional[Callable] = None

        self._start: Optional[float] = None
        self._end: Optional[float] = None

        if self.sync_gpu:  # Removes 'torch' dependency unless we want to sync gpu
            import torch
            self._sync_fn = torch.cuda.synchronize

    def __repr__(self) -> str:
        """Convert class constructor into string representation."""
        sig = inspect.signature(self.__init__)
        kwargs = {key: getattr(self, key) for key in sig.parameters if hasattr(self, key)}
        s = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        return f'{self.__class__.__qualname__}({s})'

    def __str__(self) -> str:
        """Convert into string representation."""
        return f'{self.name}: {self.elapsed} {self._units}'

    def __enter__(self) -> 'Timer':
        """Start timer and sync GPU."""
        if self.sync_gpu:
            self._sync_fn()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End timer and sync GPU."""
        if self.sync_gpu:
            self._sync_fn()
        self._end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Time taken between enter and exit."""
        assert self._start, '`Timer` has not begun'
        assert self._end, '`Timer` has not finished'
        time_taken = self._sf * (self._end - self._start)
        return round(time_taken, self.precision)


class MultiLevelTimer:
    """Context manager Timer capable of being nested across multiple levels.

    NOTE: We use the *instance* of this class as a context manager, not the class itself (see examples).

    Timers are stored as a dict, mapping labels to (depth, start, end, elapsed).
    In order to allow for the nesting of these timers, we keep track of what timers are active (effectively, a stack).
    On __exit__ we pop the most recent label and end that timer.

    Attributes:
    :param name: (str) Global Timer name.
    :param as_ms: (bool) If `True`, store time as `'milliseconds`', otherwise `seconds`.
    :param sync_gpu: (bool) If `True`, ensure that GPU is synced on Timer enter and exit.
    :param precision: (int) Number of decimal places to print.

    Examples:
    ```
    timer = MultiLevelTimer(name='MyTimer', as_ms=True, precision=4)

    with timer('OuterLevel'):
        time.sleep(2)
        with timer('InnerLevel'):
            time.sleep(1)

    print(timer)

    ==>
    MyTimer
        OuterLevel: 3002.3414 ms
            InnerLevel: 1000.7601 ms
    ```

    Levels can also be named automatically
    ```
    timer = MultiLevelTimer(name='MyTimer')

    with timer:
        time.sleep(2)

    print(timer)

    ==>
    MyTimer
        Level1: 2.002093 s
    ```
    """
    def __init__(self, name: str = 'Timer', as_ms: bool = False, sync_gpu: bool = False, precision: int = 6) -> None:
        self.name: str = name
        self.as_ms: bool = as_ms
        self.sync_gpu: bool = sync_gpu
        self.precision: int = precision

        self.depth: int = 0  # Nesting level of the current active timer.
        self._sf: int = 1000 if self.as_ms else 1
        self._units: str = 'ms' if self.as_ms else 's'
        self._sync_fn: Optional[Callable] = None

        self._label: Optional[str] = None  # Tag for the current active timer.
        self._active: list[str] = []  # Stack of active timers.
        self._data: dict[str, TimerData] = {}

        if self.sync_gpu:
            # Removes 'torch' dependency unless we want to sync gpu
            import torch
            self._sync_fn = torch.cuda.synchronize

    def __repr__(self) -> str:
        """Convert class constructor into string representation."""
        sig = inspect.signature(self.__init__)
        kwargs = {key: getattr(self, key) for key in sig.parameters if hasattr(self, key)}
        s = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        return f'{self.__class__.__qualname__}({s})'

    def __str__(self) -> str:
        """Convert into string representation."""
        s = [self.name]
        s += ['\t'*v['depth'] + f'{k}: {v["elapsed"]} {self._units}' for k, v in self]
        return '\n'.join(s)

    def __getitem__(self, label: str) -> TimerData:
        """Return timer info for the given label."""
        return self._data[label]

    def __iter__(self) -> Generator[tuple[str, TimerData], None, None]:
        """Iterate through all timers as (`label`, `timer`)"""
        for k in self._data:
            yield k, self[k]

    def __call__(self, label: str) -> 'MultiLevelTimer':
        """Required to call a `Timer` instance in a context manager and create a new label."""
        self._label = label
        return self

    def __enter__(self) -> 'MultiLevelTimer':
        """Context manager entry point."""
        self.depth += 1

        label, self._label = self._label, None  # Reset
        label = label or f'Level{self.depth}'  # Autogenerated label
        if label in self._data:
            raise KeyError(f"Duplicate Timer key: {label}")

        if self.sync_gpu: self._sync_fn()

        self._active.append(label)
        self._data[label] = {
            'depth': self.depth,
            'start': time.perf_counter(),  # Start timer
            'end': None,
            'elapsed': None,
        }
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point."""
        assert self._active, "What are you doing here??"
        label = self._active.pop()
        timer = self._data[label]

        if self.sync_gpu: self._sync_fn()

        timer['end'] = time.perf_counter()
        timer['elapsed'] = round(self._sf * (timer['end'] - timer['start']), self.precision)
        self.depth -= 1

    def reset(self) -> None:
        """Delete all existing `Timer` data."""
        if self._active: raise RuntimeError(f"Attempt to reset Timer while active: {self._active}")
        self._data = {}

    def copy(self) -> 'MultiLevelTimer':
        """Return a deep copy of the timer."""
        return copy.deepcopy(self)

    def to_dict(self, key: str = 'elapsed') -> dict:
        """Return a dict containing only the data for the specified key."""
        return {label: data[key] for label, data in self}

    @staticmethod
    def mean_elapsed(timers: Sequence['MultiLevelTimer']) -> Union[Sequence, dict[str, float]]:
        """Return the average elapsed time for each label in a list of timers."""
        if not timers: return timers

        data = {}
        for t in timers:
            for k, v in t:
                if k in data: data[k].append(v['elapsed'])
                else:         data[k] = [v['elapsed']]

        data = {k: sum(v)/len(v) for k, v in data.items()}
        return data
