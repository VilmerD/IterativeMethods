import scipy.sparse as sp
import numpy as np


def interval(n):
    return np.arange(1, n + 1) / (n + 1)


def T(n):
    return (n + 1) ** 2 * sp.csr_matrix(sp.diags([1, -2, 1], [-1, 0, 1], (n, n)))


def source(n):
    x = interval(n)
    return 2 * np.ones((n, )) - np.sin(x*(1 - x))


def F(u):
    n = len(u)
    return T(n).dot(u) + source(n) + np.sin(u)


def J(u):
    n = len(u)
    return T(n) + sp.csr_matrix(sp.diags(np.cos(u)))