#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import math

from functools import wraps
from inspect   import signature

# define consistent linear math that stick with `np.array` (rather than
# `np.matrix`) => this will mean that we're sticking with the "minimal" data
# type for vector data. NOTE: this might cause a performance hit due to
# changing data types.
mat_to_a = lambda a    : np.squeeze(np.asarray(a))
matmul_a = lambda a, b : mat_to_a(np.dot(a, b))



def resid(A, *args, **kwargs):
    if isinstance(A, np.matrix):
        A_op = lambda x: matmul_a(A, x)
        return resid_kernel(A_op, *args, **kwargs)
    else:
        return resid_kernel(A, *args, **kwargs)



def resid_kernel(A, x, b):
    return np.array(
        [np.linalg.norm(A(xi)-b) for xi in x]
    )



def prob_norm(b):
    # Compute 2-norm of RHS(for scaling RHS input to network)
    b_flat     = np.reshape(b, (1, -1), order='F').squeeze(0)
    b_norm     = np.linalg.norm(b_flat)
    b_Norm_max = np.max(b / b_norm)

    return b_norm, b_Norm_max



# mathematical indices for python
cidx   = lambda i: i-1  # c-style index from math-style index
midx   = lambda i: i+1  # math-style index from c-style index
mrange = lambda n: range(1, n + 1)
