import numpy as np
import timemory

from timemory.component import WallClock

from PyGMRES.compiler import RC, Decorated, get_undecorated_fn

RC().enable_numba = True
from PyGMRES import linop


laplace_1d_dirichlet_uncompiled = get_undecorated_fn("laplace_1d_dirichlet")

b = np.zeros((1000,))
b[100:200] = 2

# Call function once to compile
linop.laplace_1d_dirichlet(b, 1, -1)

wc = WallClock("1D Numba")
wc.start()
linop.laplace_1d_dirichlet(b, 1, -1)
wc.stop()
print(f"1D Numba = {wc.get()}")

wc = WallClock("1D Python")
wc.start()
laplace_1d_dirichlet_uncompiled(b, 1, -1)
wc.stop()
print(f"1D Python = {wc.get()}")

timemory.finalize()