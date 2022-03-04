from functools import wraps
from inspect   import signature

from . import RC, Decorated

RC().lock()

if RC().enable_numba:
    import numba
    from numba.typed import List
else:
    List = list()


def jit(**kwargs):
    """
    Conditional Numba compiler decorator that invokes the compiler iff
    RC().enable_numba = True when decorator is invoked (i.e. when the decorated
    function is first defined.)
    """

    def noop(func):
        return func

    def op(func):
        _op = wraps(func)(
            numba.jit(**kwargs)(
                func
            )
        )
        _op.__signature__ = signature(func)
        Decorated().add(func)
        return _op

    if RC().enable_numba:
        return op
    else:
        return noop