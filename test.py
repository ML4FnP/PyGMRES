#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyGMRES import compiler


@compiler.compile()
def test1():
    print("hi")

test1()

compiler.RC().enable_numba = False

@compiler.compile()
def test2():
    print("ho")

test1()
test2()
