import numpy as np
from numpy import concatenate, tile, ones, zeros, array, arange
from scipy import sparse
from scipy.sparse import diags, csr_matrix, spdiags


def poisson_matrix_1d(n):
    if n > 1:
        z = ones(n)
        return csr_matrix((n + 1) ** 2 * spdiags([z, -2*z, z], [-1, 0, 1], n, n))
    else:
        return array([-8])


def poisson_matrix_2d(n):
    if n > 1:
        q = poisson_matrix_1d(n)
        design = tile(np.concatenate((q.diagonal(1), [0])), n)
        design = concatenate(([design[0]], design))
        mid = 2*q.diagonal(0).repeat(n)
        sub = ones(n ** 2) * design[0]
        return sparse.csr_matrix(spdiags([sub, design[1:], mid, design[:-1], sub], [-n, -1, 0, 1, n], n ** 2, n ** 2))
    else:
        return array([-8])


def poisson_r(n):
    x = arange(1, n + 1) / (n + 1)
    center_diagonal = concatenate(([-1], - 2 * ones(n-1)))
    t = (n + 0.5) ** 2 * diags((ones(n-1), center_diagonal, ones(n-1)), (-1, 0, 1))

    center_s = concatenate(([-1], zeros(n-1)))
    s = (n + 0.5) * diags((-ones(n-1), center_s, ones(n-1)), (-1, 0, 1)) / 2
    r = diags(1 / x, 0)
    return csr_matrix(- r * s - t)


def interpolator2d(v):
    n = int(len(v) ** 0.5)
    v = v.reshape(n, n)

    nn = 2 * n + 1
    v_new = np.zeros((nn, nn))
    u = zeros((n + 2, n + 2))
    u[1:-1, 1:-1] = v
    for k in np.arange(-1, n):
        for j in np.arange(-1, n):
            v_new[2 * j + 1, 2 * k + 1] = u[j + 1, k + 1]
            v_new[2 * j + 2, 2 * k + 1] = (u[j + 2, k + 1] + u[j + 1, k + 1]) / 2
            v_new[2 * j + 2, 2 * k + 2] = (u[j + 2, k + 2] + u[j + 2, k + 1] + u[j + 1, k + 2] + u[j + 1, k + 1]) / 4
            v_new[2 * j + 1, 2 * k + 2] = (u[j + 1, k + 2] + u[j + 1, k + 1]) / 2
    return v_new.reshape(nn ** 2)


def restrictor2d(v):
    n = int((len(v) ** 0.5))
    v = v.reshape((n, n))

    nn = int((n - 1) / 2)
    v_new = np.zeros((nn, nn))
    for k in np.arange(0, nn):
        for j in np.arange(0, nn):
            v_new[j, k] = (v[2 * j, 2 * k] + v[2 * j + 2, 2 * k] + v[2 * j, 2 * k + 2] + v[2 * j + 2, 2 * k + 2]
                           + 2 * (v[2 * j, 2 * k + 1] + v[2 * j + 1, 2 * k + 2] + v[2 * j + 2, 2 * k + 1] +
                                  v[2 * j + 1, 2 * k]) + 4 * v[2 * j + 1, 2 * k + 1]) / 16
    return v_new.reshape(nn ** 2)
