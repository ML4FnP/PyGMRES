#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyGMRES.compiler import RC, Decorated

RC().enable_numba = True
from PyGMRES.compiler.test import jit

@jit()
def test1():
    print("hi")

@jit(nogil=True, nopython=True)
def test2():
    print("ho")

test1()
test2()
print(Decorated().record)

undecorated_test1 = next(Decorated().has_name("test1")).func
undecorated_test2 = next(Decorated().has_name("test2")).func

undecorated_test1()
undecorated_test2()