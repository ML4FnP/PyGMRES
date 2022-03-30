# PyGMRES

An all-python implementation of the GMRES algorithm. Numba acceleration is
available (and enabled by default).

## Using PyGMRES

The `GMRES` solves the linear `b = A(x)` using:
```
PyGMRES.gmres.GMRES(A, b, x0, e, max_iter, restart=None, debug=False)
```
where `A` is a linear operator, `x0` is the initial guess for the solution `x`,
`e > |A(x) - b|` is the tolerance, `max_iter` is the maximum number of
iterations -- if `GMRES` does not converge on a solution, it is stopped after
`max_iter`.

The PyGMRES solver `GMRES` can be run in two modes:
1. Regular mode: when `restart = None` the full Krylov subspace is used to solve
the problem. This mode is guaranteed to find a solution (if one exists), but
scales poorly.
2. Restarted mode: when `restart = N` then the Krylov subspace used by the
solver is rebuilt every `N` steps. This mode is not guaranteed to find a
solution for certain intial conditions, but scales well.

## Discrete Laplace Operators

PyGMRES provides implementations of the 1D and 2D discrete laplace operators in
the `PyGMRES.linop` module:
1. 1D laplacian: `laplace_1d`, `laplace_1d_dirichlet`, `laplace_1d_constextrap`,
and `laplace_1d_extrap` which provide periodic, dirichlet, constant
extrapolation, and linear extrapolation respectively.
2. 2D laplacian: `laplace_2d`, `laplace_2d_dirichlet`, `laplace_2d_constextrap`,
and `laplace_2d_extrap` which provide periodic, dirichlet, constant
extrapolation, and linear extrapolation respectively.

## Numba Compiler

The `PyGMRES.compiler` module provides the standard Numba JIT compiler
decorator. All operators and the GMRES solver are compiled when their host
modules are first imported. The compiler can be enabled, or disabled using the
`PyGMRES.compiler.RC` class. For example this will enable compilation (on by
default):

```python
from PyGMRES.compiler import RC
RC().enable_numba = True
```

Ance any module compiles any of its functions, `RC().enable_numba` cannot be
changed. `RC()` is intended for those situations where Numba cannot be installed
on a target system. If Numba is enabled, the originial (non-compiled) functions
are avaialable using `get_undecorated_fn(name)`, for example:

```python
from PyGMRES.compiler import get_undecorated_fn
laplace_1d_uncompiled = get_undecorated_fn("laplace_1d_dirichlet")
```

## Helper Functions

PyGMRES preovides helper functions for common tasks
### Computing the Residuals

`PyGMRES.linop.resid` computes the residual `A(x) - b`, for example

```python
from PyGMRES.linop import laplace_2d_extrap, resid
r_laplace_2d = resid(A, x, b)
```

## Example

The following code solved the 2D Laplace equation with von Neumann (constant
derivative) boundary conditions, along with the residuals of each iteration:

```python
import numpy  as np
from PyGMRES.gmres import GMRES
from PyGMRES.linop import laplace_2d_extrap, resid

# Use odd dimension because we want to put a point-source in the center.
Nx = 81
Ny = 81

# Set up the RHS
b = np.zeros((Nx, Ny))
b[int(Nx/2), int(Ny/2)] = 1
x0 = np.zeros_like(b)

# Tolerance of the funal results
tol = 1e-10
# Note: here we chose nmax_iter to be << N[x,y] -- if not, then this is
# basically the non-restarted version of GMRES
nmax_iter = 10
restart   = 64
# Run GMRES in debug mode => intermediate solutions are returned as an array
x_laplace_2d = GMRES(
    laplace_2d_extrap, b, x0, tol, nmax_iter, restart=restart, debug=True
)

# Compute the residuals
r_laplace_2d = resid(laplace_2d_extrap, x_laplace_2d, b)
```