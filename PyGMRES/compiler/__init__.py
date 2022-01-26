#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools   import wraps
from inspect     import signature
from collections import namedtuple

from ..util.singleton import Singleton


class RC(metaclass=Singleton):
    """
    Stores settings used by the compiler
    """
    def __init__(self):
        self._enable_numba = True

    @property
    def enable_numba(self):
        return self._enable_numba

    @enable_numba.setter
    def enable_numba(self, val):
        print("setter called")
        self._enable_numba = val


class Decorated(metaclass=Singleton):
    """
    Stores a record of decorated function
    """
    def __init__(self):
        self.descriptor = namedtuple(
            "FunctionDescriptor", ("module", "name", "signature")
        )
        self._record = set()

    @property
    def record(self):
        return self._record

    def add(self, func):
        self._record.add(
            self.descriptor(
                module=func.__module__,
                name=func.__name__,
                signature=signature(func)
            )
        )


def compile():
    """
    Conditional Numba compiler decorator that invokes the compiler iff
    RC().enable_numba = True when decorator is invoked (i.e. when the decorated
    function is first defined.)
    """

    def noop(func):
        return func

    def op(func):
        @wraps(func)
        def _op(*args, **kwargs):
            return func(*args, **kwargs)
        _op.__signature__ = signature(func)
        Decorated().add(func)
        return _op

    if RC().enable_numba:
        return op
    else:
        return noop
