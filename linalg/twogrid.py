from matrices.matricies import *
from linalg.multigrid import *

def restrictor():
    pass

def interpolator():
    pass

def n_jacobi():
    pass

def interval():
    pass

def twogrid(Afun, x0, b, pre=2, post=2):
    n = len(b)
    n_l = int((n - 1) / 2)
    A = Afun(n)

    x_tilde = n_jacobi(A, x0, b, pre, w=2/3)
    r = A.dot(x_tilde) - b
    rl_1 = restrictor(r)
    el_1 = splin.spsolve(Afun(n_l), rl_1)
    x_tilde = x_tilde - interpolator(el_1)
    x_tilde = n_jacobi(A, x_tilde, b, post, w=2/3)

    return x_tilde


def source(n):
    x = interval(n)
    return 4 * np.pi ** 2 * np.sin(np.pi * x ** 2).reshape((n,))
