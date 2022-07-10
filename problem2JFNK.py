import numpy as np
from scipy.sparse.linalg import LinearOperator, spilu

from newton.newton import JFNK, Preconditioner
from project.project_matricies import *

from linalg.smoothers import RK2Smoother
from linalg.multigrid import Multigrid, cycle, AggregateInterface1D, DefaultInterface1D, ScalableLinearOperator

import matplotlib.pyplot as plt

from time import time





def run_spec(n, height, interface=AggregateInterface1D):
    # Set up grid
    L = 2
    dx = L/n
    x = np.linspace(0, L, n+1)[1:]

    # Initial condition
    u0 = func_u0(x)
    t0, tf = 0, 1
    dt = 0.01

    # Multigrid preconditioner
    # Setup smoother
    c1, a1 = 0.99, 0.33
    hfun = lambda N: (c1 / max(u0)) * (dx / dt) / N
    smoother = RK2Smoother(a1, hfun)

    # Setup multigrid
    pre, post, gamma = 5, 5, 1
    mgpre = MultigridPreconditioner((n, n), interface, height, smoother, pre, post, gamma)

    # Setup problem
    tk, uk, U = t0, u0, [u0]
    while tk < tf:
        Func = lambda u: F(u, dt, dx, uk)
        sols, res, nits, etas = JFNK(Func, uk, M=mgpre)
        uk = sols[-1]

        tk += dt
        U.append(uk)

    # Plot
    plot = False
    if plot:
        fig, ax = plt.subplots()
        ax.plot(x, sols[-1], 'g')
        ax.plot(x, u0, 'k')
        plt.legend('u($t_0 + \Delta t$)', 'u($t_0$)')
        plt.show()
    else:
        animate(x, U)
    return


def run_specs():
    N = 2 ** np.array([8, 9, 10])

    nits = []
    residuals = []
    for n in N:
        t, r, i = run_spec(n)
        residuals.append(r)
        nits.append(i)

    lines = []
    legends = []
    colors = ['r', 'g', 'b']

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for k in range(0, len(residuals)):
        col = colors[k]
        marker = col + '--'
        lines.append(ax1.semilogy(residuals[k], marker)[0])
        legends.append('n: {}'.format(N[k]))
        ax2.plot(nits[k], col)
    plt.legend(lines, legends)
    ax1.set_ylabel('Residual')
    ax2.set_ylabel('Number of GMRES iterations')
    ax1.set_xlabel('Newton iteration number')
    plt.title('Plot over residual and number of GMRES iterations')
    plt.show()


def run_and_animate():
    L = 8
    n = 2 ** 6
    x = np.linspace(0, L, n+1)[1:]
    Nt = 500
    u = np.zeros((n ** 2, Nt + 1))
    u[:, 0] = func_u02(x)

    dt = 0.02
    dx = L/n

    print(min(u[:, 0]))
    pseudo_step = calculate_pseudo_stepsize(max(u[:, 0]), dt, L, 0.99)
    for k in range(0, Nt):
        uk = u[:, k]
        u[:, k + 1] = JFNK(lambda v: F2(v, dt, dx, uk), uk, multigrid_primer(0.33, pseudo_step))
        print("Timestep {}/{}".format(k + 1, Nt))
    np.save('2dburger.npy', u)
    animate2d('2dburger.npy')


def animate2d(name):
    L = 8
    u = np.load(name)
    n = int(u[:, 1].shape[0] ** 0.5)
    x = interval(n, length=L)
    xx, yy = np.meshgrid(x, x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for uk in u.T[:, ]:
        plt.cla()
        ax.plot_wireframe(xx, yy, uk.reshape((n, n)))
        plt.pause(0.005)
        plt.show()


def animate(x, sols):
    fig = plt.figure()
    ax = plt.axes(xlim=(min(x), max(x)), ylim=(0, 4))
    line, = ax.plot([], [])
    for uk in sols:
        line.set_data((x, uk))
        plt.pause(0.05)
        plt.show()


# run_spec(3*2**6, 4)
run_spec(3*2**6, 3, interface=AggregateInterface1D())