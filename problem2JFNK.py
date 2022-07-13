# Numpy and scipy stuff
import numpy as np

# Problems
import project.problems as problems

# Newton solvers 
from integrate.integrate import implicit_euler
from newton.precondition import MultigridPreconditioner

# Multigrid
from linalg.smoothers import RK2Smoother
import linalg.multigrid as mg

# Plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation

# Misc
from time import time
from primefac import primefac


def solve_problem(n, height=None, interface=mg.AggregateInterface1D(), smoothpara=(3, 3, 1)):
    """
    Solves the project for a grid with n unknowns
    """
    # Set up grid
    L = 2
    dx = L/n
    x = np.linspace(0, L, n+1)[1:]

    # Initial condition
    u0 = problems.ua(x)
    dt = 0.01
    t0, tf = 0, dt

    # Multigrid preconditioner
    if height is None:
        height = len(np.array([f for f in primefac(n)])==2)

    # Setup smoother
    nu = max(u0)*dt
    smoother = RK2Smoother(L, nu)

    # Setup multigrid
    pre, post, gamma = smoothpara
    mgpre = MultigridPreconditioner((n, n), interface, height, smoother, pre, post, gamma)
    # mgpre = Preconditioner()

    # Setup and solve problem
    F = lambda _, u, uk: problems.flux(u, uk, dt, dx)
    U, R, N, E = implicit_euler(F, u0, (t0, tf), dt, 1e-10, mgpre)
    return U, R[-1], N[-1], E[-1]
    

def run_specs():
    """
    Solves problem for a few grid sizes and plots the residual and
    number of inner GMRES iterations in each newton step
    """
    N = 2**np.array([7, 9, 11])

    nits = []
    residuals = []
    for n in N:
        _, r, i, _ = solve_problem(n, height=np.log2(n)-1, interface=mg.AggregateInterface1D())
        residuals.append(r)
        nits.append(i)

    lines = []
    legends = []
    colors = ['r', 'g', 'b']

    fig, ax = plt.subplots(2, 1)
    for k in range(0, len(residuals)):
        col = colors[k]
        marker = col + 'o--'
        l0 = ax[0].semilogy(residuals[k], marker)[0]
        l1 = ax[1].plot(nits[k], marker)
        lines.append(l0)
        legends.append('n: {}'.format(N[k]))

    plt.legend(lines, legends)

    # Set labels
    ax[0].set_ylabel('Residual')
    ax[1].set_ylabel('GMRES iterations')
    ax[1].set_xlabel('Newton iteration')

    # Set tickmarks
    max_newton = np.max([len(x) for x in residuals])
    max_gmres = np.max([max(x) for x in nits])
    xrange, yrange = np.arange(0, max_newton), np.arange(0, max_gmres + 1)
    ax[0].set_xticks(xrange)
    ax[1].set_xticks(xrange)
    ax[1].set_yticks(yrange)

    ax[0].grid(True)
    ax[1].grid(True)
    plt.show()

def animate(x, sols):
    """
    Animates solution!
    """
    fig, ax = plt.subplots()
    xdata, ydata = x, sols[0]
    ln, = plt.plot(xdata, ydata, 'g')

    def init():
        ax.set_xlim((np.min(x), np.max(x)))
        ax.set_ylim((np.min(sols), np.max(sols)))
        return ln,

    def update(frame):
        xdata = x
        ydata = sols[frame]
        ln.set_data(xdata, ydata)
        return ln,

    frames = np.arange(0, len(sols))

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

    plt.show()

def grid():
    """ Investigates the number of pre- and post smoothing steps """
    exponent = 8
    n = 2**exponent
    height = 8

    pmax = 10
    pre = np.arange(1, pmax)
    pst = np.arange(1, pmax)
    gam = 1

    PRE, PST = np.meshgrid(pre, pst)
    M, N, G = [], [], []
    for re, st in zip(PRE.flatten(), PST.flatten()):
        _, r, i, _ = solve_problem(n, height, smoothpara=(re, st, gam))

        newtons = len(i)
        gmress = sum(i)
        matvecs = gmress*(1 + height*(re + st) + 2**(exponent - height)) + newtons

        M.append(matvecs)
        N.append(newtons)
        G.append(gmress)
    Xlist = (np.array(X).reshape(PRE.shape) for X in (M, N, G))
    names = ('Matvecs', 'Newton', 'GMRES')

    # Plot results
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(14, 4)

    cmap = plt.colormaps['magma']
    for ax, X, name in zip(axs, Xlist, names):
        levels = ticker.MaxNLocator(nbins=len(np.unique(X))).tick_values(X.min(), X.max())
        norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        cf = ax.pcolormesh(pre, pst, X, cmap=cmap, norm=norm)
        ax.set_title(name)
        fig.colorbar(cf, ax=ax)

    axs[1].set_xlabel('Pre')
    axs[0].set_ylabel('Post')

    fig.tight_layout()
    plt.show()

grid()
