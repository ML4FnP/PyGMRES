#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyGMRES.compiler import compile, RC, Decorated

RC().enable_numba = False

@compile()
def test1():
    print("hi")

RC().enable_numba = True

@compile()
def test2():
    print("ho")

test1()
test2()
print(Decorated().record)