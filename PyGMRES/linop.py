import numpy as np
from .compiler.numba import jit


# Here we define consistent linear math that stick with `np.array` (rather than
# `np.matrix`) => this will mean that we're sticking with the "minimal" data
# type for vector data. NOTE: this might cause a performance hit due to changing
# data types.

@jit(nogil=True, nopython=True)
def mat_to_a(a):
    return np.asarray(a)


@jit(nogil=True, nopython=True)
def matmul_a(a, b):
    return mat_to_a(np.dot(a, b))


@jit(nogil=True, nopython=True)
def resid(A, x, b):
    return np.array(
        [np.linalg.norm(A(xi)-b) for xi in x]
    )


@jit(nogil=True, nopython=True)
def laplace_1d(x, i):
    return 2*x[i] - x[i-1] - x[i+1]


@jit(nogil=True, nopython=True)
def laplace_1d_dirichlet(x_in, xlo, xhi):
    '''
    laplace_1d_dirichlet(x_in, xlo, xhi)

    Applies laplace operator as a stencil operation for a N-cell 1D grid, for
    given boundary conditionds.
    '''
    N, = x_in.shape
    x_pad = np.zeros((N + 2,))
    x_pad[1:N+1] = x_in[:]

    x_pad[0]   = xlo
    x_pad[N+1] = xhi

    Ax = np.zeros_like(x_in)
    for ix, in np.ndindex(Ax.shape):
        Ax[ix] = laplace_1d(x_pad, ix+1)

    return Ax


@jit(nogil=True, nopython=True)
def laplace_1d_constextrap(x_in):
    '''
    laplace_1d_constextrap(x_in)

    Applies laplace operator as a stencil operation for a N-cell 1D grid, for
    given boundary conditionds.
    '''
    N, = x_in.shape
    x_pad = np.zeros((N + 2,))
    x_pad[1:N+1] = x_in[:]

    x_pad[0]   = x_pad[1]
    x_pad[N+1] = x_pad[N]

    Ax = np.zeros_like(x_in)
    for ix, in np.ndindex(Ax.shape):
        Ax[ix] = laplace_1d(x_pad, ix+1)

    return Ax


@jit(nogil=True, nopython=True)
def laplace_1d_extrap(x_in):
    '''
    laplace_1d_constextrap(x_in)

    Applies laplace operator as a stencil operation for a N-cell 1D grid, for
    given boundary conditionds.
    '''
    N, = x_in.shape
    x_pad = np.zeros((N + 2,))
    x_pad[1:N+1] = x_in[:]

    x_pad[0] = x_pad[1] - (x_pad[2] - x_pad[1])
    x_pad[N+1] = x_pad[N] + (x_pad[N] - x_pad[N-1])

    Ax = np.zeros_like(x_in)
    for ix, in np.ndindex(Ax.shape):
        Ax[ix] = laplace_1d(x_pad, ix+1)

    return Ax


@jit(nogil=True, nopython=True)
def laplace_2d(x, i, j):
    return (-4*x[i, j] + x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])


@jit(nogil=True, nopython=True)
def laplace_2d_dirichlet(x_in, xlo, xhi, ylo, yhi):
    '''
    laplace_2d_dirichlet(x_in, xlo, xhi, ylo, yhi)

    Applies laplace operator as a stencil operation for a N-cell 2D grid, for
    given boundary conditionds.
    '''
    Nx, Ny = x_in.shape
    x_pad = np.zeros((Nx + 2, Ny + 2))
    x_pad[1:Nx+1, 1:Ny+1] = x_in[:, :]

    x_pad[0,    :] = xlo
    x_pad[Nx+1, :] = xhi
    x_pad[:,    0] = ylo
    x_pad[:, Ny+1] = yhi

    Ax = np.zeros((Nx, Ny))
    for ix, iy in np.ndindex(Ax.shape):
        Ax[ix, iy] = laplace_2d(x_pad, ix+1, iy+1)

    return Ax


@jit(nogil=True, nopython=True)
def laplace_2d_extrap(x_in):
    '''
    laplace_2d_extrap(x_in)

    Applies laplace operator as a stencil operation for a N-cell 2D grid, for
    given boundary conditionds.
    '''
    Nx, Ny = x_in.shape
    x_pad = np.zeros((Nx + 2, Ny + 2))
    x_pad[1:Nx+1, 1:Ny+1] = x_in[:, :]

    x_pad[0,    :] = x_pad[1,  :] - (x_pad[2,  :] - x_pad[1,    :])
    x_pad[Nx+1, :] = x_pad[Nx, :] + (x_pad[Nx, :] - x_pad[Nx-1, :])
    x_pad[:,    0] = x_pad[:,  1] - (x_pad[:,  2] - x_pad[:,    1])
    x_pad[:, Ny+1] = x_pad[:, Ny] + (x_pad[:, Ny] - x_pad[:, Ny-1])

    Ax = np.zeros((Nx, Ny))
    for ix, iy in np.ndindex(Ax.shape):
        Ax[ix, iy] = laplace_2d(x_pad, ix+1, iy+1)

    return Ax


@jit(nogil=True, nopython=True)
def laplace_2d_constextrap(x_in):
    '''
    laplace_2d_constextrap(x_in)

    Applies laplace operator as a stencil operation for a N-cell 2D grid, for
    given boundary conditionds.
    '''
    Nx, Ny = x_in.shape
    x_pad = np.zeros((Nx + 2, Ny + 2))
    x_pad[1:Nx+1, 1:Ny+1] = x_in[:, :]

    x_pad[0,    :] = x_pad[1,  :]
    x_pad[Nx+1, :] = x_pad[Nx, :]
    x_pad[:,    0] = x_pad[:,  1]
    x_pad[:, Ny+1] = x_pad[:, Ny]

    Ax = np.zeros((Nx, Ny))
    for ix, iy in np.ndindex(Ax.shape):
        Ax[ix, iy] = laplace_2d(x_pad, ix+1, iy+1)

    return Ax