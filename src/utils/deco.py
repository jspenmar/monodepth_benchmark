import inspect
import random
import types
from functools import partial, wraps
from typing import Any, Callable, Optional, Sequence, Union

__all__ = ['opt_args_deco', 'delegates', 'map_container', 'retry_new_on_error']


def opt_args_deco(deco: Callable) -> Callable:
    """Meta-decorator to make implementing of decorators with optional arguments more intuitive

    Recall: Decorators are equivalent to applying functions sequentially
        >>> func = deco(func)

    If we want to provide optional arguments, it would be the equivalent of doing:
        >>> func = deco(foo=10)(func)
    I.e. in this case, deco is actually a function that RETURNS a decorator (a.k.a. a decorator factory)

    In practice, this is typically implemented with two nested functions as opposed to one.
    Also, the "factory" must always be called, "func = deco()(func)", even if no arguments are provided.
    This is ugly, obfuscated and makes puppies cry. No one wants puppies to cry.

    This decorator "hides" one level of nesting by using the 'partial' function.
    If no optional parameters are provided, we proceed as a regular decorator using the default parameters.
    If any optional kwargs are provided, this returns the decorator that is then applied to the function (this is
    equivalent to the "deco(foo=10)" portion of the second example).

    Example (before):
    ```
        def stringify(func=None, *, prefix='', suffix=''):
            if func is None:
                return partial(stringify, prefix=prefix, suffix=suffix)

            @wraps(func)
            def wrapper(*args, **kwargs):
                out = func(*args, **kwargs)
                return f'{prefix}{out}{suffix}'
            return wrapper
    ```

    Example (after):
    ```
        @opt_args_deco
        def stringify(func, prefix='', suffix=''):
            @wraps(func)
            def wrapper(*args, **kwargs):
                out = func(*args, **kwargs)
                return f'{prefix}{out}{suffix}'
            return wrapper
    ```

    :param deco: (Callable) Decorator function with optional parameters to wrap.
    :return: (Callable) If `func` is provided: decorated func, otherwise: decorator to apply to `func`.
    """
    @wraps(deco)
    def wrapper(f: Optional[Callable] = None, **kwargs) -> Callable:
        # If only optional arguments are provided --> return decorator
        if f is None: return partial(deco, **kwargs)

        # Soft-enforcing that we provide the optional arguments as keyword only
        if not isinstance(f, (types.FunctionType, types.MethodType)):
            raise TypeError(f"Positional argument must be a function or method, got {f} of type {type(f)}")

        # Pass kwargs to allow for programmatic decorating --> return decorated function
        return deco(f, **kwargs)
    return wrapper


def delegates(to: Optional[Callable] = None, keep: bool = False):
    """From https://www.fast.ai/2019/08/06/delegation/
    Decorator to replace `**kwargs` in signature with params from `to`.

    This can be used to decorate either a class
    ```
        @delegates()
        class Child(Parent): ...
    ```
    or a function
    ```
        @delegates(parent)
        def func(a, **kwargs): ...
    ```

    :param to: (Callable) Callable containing the params to copy
    :param keep: (bool) If `True`, keep `**kwargs` in the signature.
    :return: (Callable) The decorated class or function with the updated signature.
    """
    def wrapper(f: Union[type, Callable]) -> Callable:
        to_f, from_f = (f.__base__.__init__, f.__init__) if to is None else (to, f)
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)

        args = sigd.pop('args', None)
        if args:
            sigd2 = {k: v for k, v in inspect.signature(to_f).parameters.items()
                     if v.default == inspect.Parameter.empty and k not in sigd}
            sigd.update(sigd2)

        kwargs = sigd.pop('kwargs', None)
        if kwargs:
            sigd2 = {k: v for k, v in inspect.signature(to_f).parameters.items()
                     if v.default != inspect.Parameter.empty and k not in sigd}
            sigd.update(sigd2)

        if keep and args:   sigd['args'] = args
        if keep and kwargs: sigd['kwargs'] = kwargs

        from_f.__signature__ = sig.replace(parameters=list(sigd.values()))
        return f

    return wrapper


def map_container(f: Callable) -> Callable:
    """Decorator to recursively apply a function to arbitrary nestings of `dict`, `list`, `tuple` & `set`

    NOTE: `f` can have an arbitrary signature, but the first arg must be the item we want to apply `f` to.

    Example:
    ```
        @map_apply
        def square(n, bias=0):
            return (n ** 2) + bias

        x = {'a': [1, 2, 3], 'b': 4, 'c': {1: 5, 2: 6}}
        print(map_apply(x))

        ===>
        {'a': [1, 4, 9], 'b': 16, 'c': {1: 25, 2: 36}}

        print(map_apply(x, bias=2))

        ===>
        {'a': [3, 6, 11], 'b': 18, 'c': {1: 27, 2: 38}}
    ```
    """
    @wraps(f)
    def wrapper(x: Any, *args, **kwargs) -> Any:
        if isinstance(x, dict):
            return {k: wrapper(v, *args, **kwargs) for k, v in x.items()}

        elif isinstance(x, list):
            return [wrapper(v, *args, **kwargs) for v in x]

        elif isinstance(x, tuple):
            return tuple(wrapper(v, *args, **kwargs) for v in x)

        elif isinstance(x, set):
            return {wrapper(v, *args, **kwargs) for v in x}

        else:  # Base case, single item
            return f(x, *args, **kwargs)

    return wrapper


@opt_args_deco
def retry_new_on_error(__getitem__: Callable,
                       exc: Union[BaseException, Sequence[BaseException]] = Exception,
                       silent: bool = False,
                       max: Optional[int] = None,
                       use_blacklist: bool = False) -> Callable:
    """Decorator to wrap a BaseDataset __getitem__ function, and retry a different index if there is an error.

    The idea is to provide a way of ignoring missing/corrupt data without having to blacklist files,
    change number of items and do "hacky" workarounds.
    Obviously, the less data we have, the less sense this decorator makes, since we'll start duplicating more
    and more items (although if we're augmenting our data, it shouldn't be too tragic).
    Obviously as well, for debugging/evaluation it probably makes more sense to disable this decorator.

    NOTE: This decorator assumes we follow the BaseDataset format
        - We return three dicts (x, y, meta)
        - Errors are logged in meta['errors']
        - A 'log_timings' flag indicates the presence of a 'MultiLevelTimer' in self.timer

    :param __getitem__: (Callable) Dataset `__getitem__` method to decorate.
    :param exc: (tuple|Exception) Expected exceptions to catch and retry on.
    :param silent: (bool) If `False`, log error info to `meta`.
    :param max: (None|int) Maximum number of retries for a single item.
    :param use_blacklist: (bool) If `True`, keep a list of items to avoid.
    :return: (tuple[dict]) x, y, meta returned by `__getitem__`.
    """
    n = 0
    blacklist = set()

    # Multiple exceptions must be provided as tuple
    exc = exc or tuple()
    if isinstance(exc, list):
        exc = tuple(exc)

    @wraps(__getitem__)
    def wrapper(cls, item):
        nonlocal n

        try:
            x, y, m = __getitem__(cls, item)
            if not silent and 'errors' not in m:
                m['errors'] = ''
        except exc as e:
            n += 1
            if max and n >= max:
                raise RuntimeError('Exceeded max retries when loading dataset item...')

            if use_blacklist: blacklist.add(item)
            if cls.log_time: cls.timer.reset()

            new = item
            while new == item or new in blacklist:  # Force new item
                new = random.randrange(len(cls))

            x, y, m = wrapper(cls, new)
            if not silent: m['errors'] += f'{" - " if m["errors"] else ""}{(item, e)}'

        n = 0  # Reset!
        return x, y, m
    return wrapper
