#!/usr/bin/env python
# -*- coding: utf-8 -*-

from inspect     import signature
from collections import namedtuple

from ..util.singleton import Singleton


class RC(metaclass=Singleton):
    """
    Stores settings used by the compiler
    """
    def __init__(self):
        self._enable_numba = True
        self._lock = False

    @property
    def enable_numba(self):
        return self._enable_numba

    @enable_numba.setter
    def enable_numba(self, val):
        if not self._lock:
            self._enable_numba = val
        else:
            raise RuntimeError(
                "Cannot set enable_numba after compiler has been loaded"
            )

    def lock(self):
        self._lock = True


class Decorated(metaclass=Singleton):
    """
    Stores a record of decorated function
    """
    def __init__(self):
        self.descriptor = namedtuple(
            "FunctionDescriptor", ("module", "name", "signature", "func")
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
                signature=signature(func),
                func=func
            )
        )

    def has_name(self, name):
        return filter(lambda e:e.name==name, self._record)


def get_undecorated_fn(name):
    return next(Decorated().has_name(name)).func