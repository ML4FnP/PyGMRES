#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyGMRES.compiler import RC, Decorated

RC().enable_numba = True
from PyGMRES.compiler.numba import jit

@jit()
def test1():
    print("hi")

@jit(nogil=True, nopython=True)
def test2():
    print("ho")

test1()
test2()
print(Decorated().record)