import scipy.sparse.linalg as splin
import scipy.sparse.extract as ex

from matrices.nonlinear_matricies import *
from newton.newton import *
from newton.preconditioners import *

import matplotlib.pyplot as plt
from time import time


def test1():
    n = 999
    res, sols, etas, nits = NK(F, J, n, 0.999, M=ilu)

    breakpoints = np.cumsum(np.array(nits) + 1) - 1
    fig, ax = plt.subplots()
    plt.semilogy(res)
    plt.show()


def spectrum2():
    n = 99
    res, sols, etas, nits = NK(F, J, n, 0.1, 15, M=None)

    res, sols_gs, etas, nits = NK(F, J, n, 0.1, 15, M=gauss_seidel)

    res, sols_ilu, etas, nits = NK(F, J, n, 0.1, 15, M=ilu)

    A = J(sols[0])
    w1, j = splin.eigs(A, k=n - 2)
    y1 = w1.imag
    x1 = w1.real
    ax1 = plt.subplot(131)
    plt.plot(x1, y1, '.')

    A = J(sols_gs[0])
    gs = gauss_seidel(A)(np.eye(n))
    A_gs = A.dot(gs)
    w2, j = splin.eigs(A_gs, k=n - 2)
    y2 = w2.imag
    x2 = w2.real
    ax2 = plt.subplot(132)
    plt.plot(x2, y2, '.')
    ax2.set_xlim(1 - 3 * 10 ** -14, 1 + 3 * 10 ** -14)

    A = J(sols_ilu[0])
    ILU = ilu(A)(np.eye(n))
    A_ilu = A.dot(ILU)
    w3, j = splin.eigs(A_ilu, k=n - 2)
    y3 = w3.imag
    x3 = w3.real
    ax3 = plt.subplot(133)
    plt.plot(x3, y3, '.')
    ax3.set_xlim(1 - 3*10**-14, 1 + 3*10**-14)

    plt.suptitle('Spectrum of the Matricies')
    ax1.set_title('Without preconditioner')
    ax2.set_title('Gauss-Seidel')
    ax3.set_title('ILU')
    plt.show()


def residuals():
    n = 999
    res, sols, etas, nits = NK(F, J, n, 0.1, M=None)

    res_gs, sols_gs, etas, nits_gs = NK(F, J, n, 0.1, M=gauss_seidel)

    res_ilu, sols_ilu, etas, nits_ilu = NK(F, J, n, 0.1, M=ilu)

    ax1 = plt.subplot(231)
    plt.semilogy(range(0, len(res)), np.abs(res))
    plt.subplot(234)
    plt.plot(range(0, len(nits)), nits, '.')

    ax2 = plt.subplot(232)
    plt.semilogy(range(0, len(res_gs)), np.abs(res_gs))
    plt.subplot(235)
    plt.plot(range(0, len(nits_gs)), nits_gs, '.')

    ax3 = plt.subplot(233)
    plt.semilogy(range(0, len(res_ilu)), np.abs(res_ilu))
    plt.subplot(236)
    plt.plot(range(0, len(nits_ilu)), nits_ilu, '.')

    plt.suptitle('Residual in GMRES (top) and number of GMRES iterations (bottom)')
    ax1.set_title('Without preconditioner')
    ax2.set_title('Gauss-Seidel')
    ax3.set_title('ILU, fill_factor = 4')
    plt.show()


def speed():
    n = 99
    n1 = 100
    avs = []
    t_ilu = []
    for k in range(0, n1):
        t1 = time()
        res_ilu, sols_ilu, etas, nits_ilu = NK(F, J, n, 0.1, M=ilu)
        t = time() - t1
        t_ilu.append(t)


test1()