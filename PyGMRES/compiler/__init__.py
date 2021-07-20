#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import wraps
from inspect   import signature



class Singleton(type):
    """
    Singleton Class type
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]



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


def compile():
    """
    """

    def noop(func):
        @wraps(func)
        def _noop(*args, **kwargs):
            print("NOOP")
            return func(*args, **kwargs)
        _noop.__signature__ = signature(func)
        return _noop


    def op(func):
        @wraps(func)
        def _op(*args, **kwargs):
            print("OP")
            return func(*args, **kwargs)
        _op.__signature__ = signature(func)
        return _op


    if RC().enable_numba:
        return op
    else:
        return noop
