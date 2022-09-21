import warnings
from typing import Callable, Optional, Sequence, Union

import torch.optim.lr_scheduler as sched

from src.typing import DataDict, ModDict, SchedDict

__all__ = ['NET_REG', 'LOSS_REG', 'DATA_REG', 'SCHED_REG', 'register']


NET_REG: ModDict = {}
LOSS_REG: ModDict = {}
DATA_REG: DataDict = {}
SCHED_REG: SchedDict = {
    'steplr': sched.StepLR,
    'exp': sched.ExponentialLR,
    'cos': sched.CosineAnnealingLR,
    'cos_warm': sched.CosineAnnealingWarmRestarts,
    'plateau': sched.ReduceLROnPlateau,
}

# Collection of registries.
_REG = {
    'net': NET_REG,
    'loss': LOSS_REG,
    'data': DATA_REG,
}

# Patterns matching class name endings to registry types.
_NAME2TYPE = {
    'Net': 'net',
    'Loss': 'loss',
    'Reg': 'loss',
    'Dataset': 'data',
}


def register(name: Union[str, Sequence[str]], type: Optional[str] = None, overwrite: bool = False) -> Callable:
    """Class decorator to build a registry of networks, losses & data available during training.

    :param name: (str|Sequence[str]) Key(s) to access class in the registry.
    :param type: (None|str) Registry to use. If `None`, guess from class name. {None, 'net', 'loss', 'data'}
    :param overwrite: (bool) If `True`, overwrite class `name` in registry `type`.
    :return:
    """
    def get_type(cls):
        """Helper to identify registry `type` from class name."""
        try:
            return next(v for k, v in _NAME2TYPE.items() if cls.__name__.endswith(k))
        except StopIteration:
            raise ValueError(f'Class matched no valid patterns. ("{cls.__name__}" vs. {set(_NAME2TYPE)})')

    def wrapper(cls):
        """Decorator adding `cls` to the specified registry."""
        # Ignore classes created in the __main__ entrypoint to avoid duplication.
        if cls.__module__ == '__main__':
            warnings.warn(f'Ignoring class "{cls.__name__}" created in the "__main__" module.')
            return cls

        ns = (name,) if isinstance(name, str) else name
        t = type or get_type(cls)
        if t not in _REG: raise TypeError(f'Invalid `type`. ("{t}" vs. {set(_REG)})')

        reg = _REG[t]
        for n in ns:
            if not overwrite and (tgt := reg.get(n)):
                raise ValueError(f'"{n}" already in "{t}" registry ({tgt} vs. {cls}). Set `overwrite=True` to overwrite it.')
            reg[n] = cls
        return cls
    return wrapper
