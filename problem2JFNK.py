import numpy as np

from newton.newton import JFNK
from newton.preconditioners import multigrid_primer
from project.project_matricies import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from time import time


def calculate_pseudo_stepsize(a, dt, L, c1):
    return lambda N: c1 * L / (a * dt * N)


def run_spec(n):
    L = 2
    x = interval(n, length=L)
    u0 = func_u0(x)
    dt = 0.1
    dx = L/n

    pseudo_step = calculate_pseudo_stepsize(max(u0), dt, L, 0.99)
    t1 = time()
    r, nits = JFNK(lambda u: F(u, dt, dx, u0), u0, M=multigrid_primer(0.33, pseudo_step))
    t2 = time() - t1
    return t2, np.array(r)/r[0], nits


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
    x = interval(n, length=L)
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


def animate(name):
    L = 2
    u = np.load(name)
    n = u[:, 1].shape[0]
    x = interval(n, length=L)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, L), ylim=(0, 4))
    line, = ax.plot([], [])
    for uk in u.T[:, ]:
        plt.cla()
        plt.plot(x, uk)
        plt.axis([0, L, 0, 4])
        plt.pause(0.03)
        plt.show()


run_spec(128)