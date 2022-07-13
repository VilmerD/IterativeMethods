import numpy as np
from integrate.integrate import implicit_euler

# Newton solvers 
from newton.newton import JFNK
from newton.precondition import MultigridPreconditioner

# Multigrid
from linalg.smoothers import RK2Smoother
import linalg.multigrid as mg

# Plotting
import matplotlib.pyplot as plt

# Problem
import project.problems as problems

def conservation_law(flux):
    def wrapper(u, uold, dt, dx, a, b, *args):
        # Flux at u
        fu = flux(u, *args)

        # Flux shifted -1
        fum1 = flux(np.roll(u, -1), *args)
        fum1[0], fum1[-1] = a(u), b(u)        # Boundary fluxes

        return u - uold + dt/dx * (fu - fum1)
    return wrapper


@problems.conservation_law_periodic
def car_flux(u, umax=100, vmax=70):
    """ Describes the flux of cars per unit length """
    return car_speed(u, umax, vmax)*u


def car_speed(u, umax, vmax):
    """ Describes car speed """
    return vmax*(1 - u/umax)


def zero(func):
    """ Gives FV values of flux """
    def wrapper(x, *args):
        fx = func(x[:-1], *args)
        fxs = func(x[1:], *args)
        return (fx + fxs)/2
    return wrapper


def zero2(func):
    def wrapper(x, *args):
        return func((x[:-1] + x[1:])/2, *args)
    return wrapper


def ua(x, umax=1, padding=0):
    u_m_padding = (x < -padding)*umax
    u_padding = (np.abs(x) < padding)*(-x + padding)/(2*padding + 1e-9)*umax
    u_p_padding = (x > padding)*0*umax
    return u_m_padding + u_padding + u_p_padding


def ub(x, umax):
    upos = umax*(x - x[0]*0.5)*(x - x[-1]*0.51)/(0 - x[0]*0.5)/(0 - x[-1]*0.51)
    return upos*((x > x[0]*0.5)*1*(x < x[-1]*0.51)*1)


def plot_peicewise_constant(ax, xx, uu, *args):
    l = []
    for x, xp1, u, up1 in zip(xx[:-1], xx[1:], uu[:-1], uu[1:]):
        lx, = ax.plot([x, xp1], [u, u], *args)
        lxp1, = ax.plot([xp1, xp1], [u, up1], *args)
        l.append((lx, lxp1))
    return l


def plot_fluxes():
    xx = np.linspace(-1, 1, 100)

    umax1, umax2, pad1, pad2 = 10, 10, 0, 0.05
    u1 = ua(xx)
    u2 = ua(xx, umax1, pad1)
    u3 = ua(xx, umax2, pad2)
    names = ('Default', f'umax={umax1}, pad={pad1}', f'umax={umax2}, pad={pad2}')
    lines = []

    fig, ax = plt.subplots(1, 1)
    for u, color in zip((u1, u2, u3), ('g', 'r', 'k')):
        l = plot_peicewise_constant(ax, xx, u, color)
        lines.append(l)

    plt.show()


def solve_problem(N):
    # Problem variables
    umax = 100
    vmax = 70

    # Set up grid
    L = 100
    # Let xn denote the nth point, where x0 = -L, x(N+1) = L
    # ie xn = -L + n*dx, and x(N + 1) = -L + (N+1)dx = L
    dx = 2*L/(N + 1)
    xx = np.linspace(-L, L, N + 1)

    # Initial condition
    # the k:th entry of u corresponds to the value in the cell
    # bounded by xk and xk+1
    u0 = problems.zero_periodic(ub)(xx[:-1], 50)

    # Integration
    dt = 2e-2
    t0, tf = 0, 500*dt

    # Smoother
    nu = np.max(car_speed(u0, umax, vmax)*u0)*dt
    smoother = RK2Smoother(2*L, nu)
    
    # Multigrid
    pre, post, gamma = 3, 3, 1
    interface = mg.AggregateInterface1D()
    mgpre = MultigridPreconditioner((N, N), interface, np.log2(N), \
        smoother, pre, post, gamma)

    # Setup and solve problem
    F = lambda _, u, uk : car_flux(u, uk, dt, dx, umax, vmax)
    U, R, N, E = implicit_euler(F, u0, (t0, tf), dt, 1e-10, mgpre)
    plot_results(xx, U)


def plot_results(xx, U):
    fig, ax = plt.subplots()
    plot_peicewise_constant(ax, xx, U[0], 'g')
    plot_peicewise_constant(ax, xx, U[-1], 'k')
    plt.show()

if __name__=='__main__':
    solve_problem(512)
