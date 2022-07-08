from Linear_Solvers.multigrid import *
from Project.project_matricies import *
from Linear_Solvers.smoothers import *
import scipy.sparse.linalg as splin
import matplotlib.pyplot as plt
import numpy as np
from time import time


def F_u(u, h):
    up = u ** 2
    return u + h * u.shape[0] / 4 * (up - np.roll(up, -1))


def F_linop(h):
    def F_wrapper(n):
        return splin.LinearOperator((n, n), lambda u: F_u(u, h))
    return F_wrapper


def run_spec(n):
    L = 2
    x = interval(n, length=L)
    u0 = func_u0(x)
    u = u0.copy()

    dt = 0.018
    fnorm = np.linalg.norm(u)

    residual0 = np.linalg.norm(F_u(u, dt) - u) / fnorm
    residuals = [residual0]

    t1 = time()
    while residuals[-1] > 10 ** -9:
        u = FAS(F_linop(dt), u, u0, RungeKutta(0.25, lambda N: 50 / N))
        residual = np.linalg.norm(F_u(u, dt) - u0) / fnorm
        residuals.append(residual)
        if abs(residuals[-1]/residuals[-2]) > 0.999:
            print("Stagnated at: {}".format(residuals[-1]))
            break
        print(int(np.log10(residual)))

    return residuals


def animate(u):
    L = 2
    n = 2 ** 8
    x = interval(n, length=L)

    Nt = 100
    fig = plt.figure()
    ax = plt.axes(xlim=(0, L), ylim=(0, 4))
    line, = ax.plot([], [])

    for k in range(0, Nt):
        plt.cla()
        plt.plot(x, u[:, k])
        plt.axis([0, L, 0, 4])
        plt.pause(0.1)
        plt.show()


def run_specs():
    N = 2 ** np.array([8, 9, 10])

    nits = []
    residuals = []
    for n in N:
        r = run_spec(n)
        residuals.append(r)

    lines = []
    legends = []
    colors = ['r', 'g', 'b']

    fig, ax1 = plt.subplots()
    for k in range(0, len(residuals)):
        col = colors[k]
        marker = col + '--'
        lines.append(ax1.semilogy(residuals[k], marker)[0])
        legends.append('n: {}'.format(N[k]))
    plt.legend(lines, legends)
    ax1.set_ylabel('Residual')
    ax1.set_xlabel('FAS iteration number')
    plt.title('Plot over residual')
    plt.show()


run_specs()
