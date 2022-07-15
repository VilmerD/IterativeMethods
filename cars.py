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
from matplotlib.animation import FuncAnimation

# Problem
import project.problems as problems

def conservation_law(flux):
    def wrapper(u, uold, dt, dx, a, *args):
        # Flux at u
        fu = flux(u, *args)

        # Flux shifted -1
        fum1 = flux(np.roll(u, 1), *args)
        fum1[0] = a(u)       # Boundary fluxes

        return u - uold + dt/dx * (fu - fum1)
    return wrapper


def car_flux(u, umax=100, vmax=70):
    """ Describes the flux of cars per unit length """
    return car_speed(u, umax, vmax)*u


def car_speed(u, umax, vmax):
    """ Describes car speed """
    return vmax*(1 - u/umax)


def car_concentration(v, umax, vmax):
    """ Returns car concentration given their speed """
    return umax*(1 - v/vmax)


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


def ua(x, umax=100, e=0):
    u1 = (x < -e)*umax
    u12pad = (np.abs(x) < e)*(x - (-e))/(2*e + 1e-9)*(0 - umax) + umax
    u2 = (x > e)*0
    return u1 + u12pad + u2


def ub(x, umax):
    upos = umax*(x - x[0]*0.5)*(x - x[-1]*0.51)/(0 - x[0]*0.5)/(0 - x[-1]*0.51)
    return upos*((x > x[0]*0.5)*1*(x < x[-1]*0.51)*1)


def uc(x, D1, u0max, vi, vmax, e):
    cc1 = car_concentration(vi, u0max, vmax)
    u1 = (x - (-e) < 0)*cc1
    u12pad = (np.abs(x) < e)*((x - (-e))/(2*e + 1e-9)*(u0max - cc1) + cc1)
    
    u2 = (x - e > 0)*(x - D1 < -e)*u0max
    u23pad = (np.abs(x - D1) < e)*((x - (D1 - e))/(2*e + 1e-9)*(0 - u0max) + u0max)

    u3 = (x - D1 > e)*(x > D1 + e)*0
    return u1 + u12pad + u2 + u23pad + u3


def plot_peicewise_constant(ax, xx, uu, *args):
    l = []
    for x, xp1, u, up1 in zip(xx[:-1], xx[1:], uu[:-1], uu[1:]):
        lx, = ax.plot([x, xp1, xp1], [u, u, up1], *args)
        l.append(lx)
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


def animate(N):
    # Problem variables
    umax = 100      # cars per unit length
    vmax = 70      # unit length per unit time

    # Set up grid
    L = 10e3        # meters

    # Let xn denote the nth point, where x0 = -L, x(N+1) = L
    # ie xn = -L + n*dx, and x(N + 1) = -L + (N+1)dx = L
    dx = 2*L/(N + 1)
    xx = np.linspace(-L, L, N + 1)

    # Initial condition
    # the k:th entry of u corresponds to the value in the cell
    # bounded by xk and xk+1
    D1, u0max, vi = L/2, 50, 60
    u0 = zero2(uc)(xx, D1, u0max, vi, vmax, (2*L)/(N/2**1))
    v0 = car_speed(u0, umax, vmax)*u0

    # Integration
    M, dt, t0 = 500, L/vmax/100, 0       # 1, 1

    # Smoother
    nu = np.max(car_speed(u0, umax, vmax)*u0)*dt
    smoother = RK2Smoother(2*L, nu)
    
    # Multigrid
    pre, post, gamma = 3, 3, 1
    interface = mg.AggregateInterface1D()
    mgpre = MultigridPreconditioner((N, N), interface, np.log2(N)-2, \
        smoother, pre, post, gamma)

    # Setup and solve problem
    a = lambda u: car_flux(car_concentration(vi, umax, vmax), umax, vmax)
    F = lambda _, u, uk : conservation_law(car_flux)(u, uk, dt, dx, a, umax, vmax)
    tk, U = t0, [u0]

    fig, axu = plt.subplots()
    arts = plot_peicewise_constant(axu, xx, u0, 'k')
    len1 = len(arts)

    timeform = 'time: {:5.2f}'
    # unitform = r' $\frac{L}{1000 \times v_{max}}$'
    unitform = ''
    txt = axu.text(1, 1, timeform.format(t0*vmax/L) + unitform, \
        horizontalalignment='right', verticalalignment='bottom', \
            transform=axu.transAxes)
    
    arts.append(txt)

    def init():
        axu.set_xlim((np.min(xx), np.max(xx)))
        axu.set_ylim((0, umax))

        axu.set_xticks([-L, 0, L])
        axu.set_yticks([0, umax])

        axu.set_xticklabels(['$-L$', '$0$', '$L$'])
        axu.set_yticklabels(['$0$', r'$u_{max}$'])
        return arts

    def update(frame):
        func = lambda u: F(tk, u, U[-1])
        sols, _, _, _ = JFNK(func, U[-1], rtol=1e-10, M=mgpre)
        uk = sols[-1]
        vk = car_speed(uk, umax, vmax)*uk

        for l, u, up1 in zip(arts[:len1], uk[:-1], uk[1:]):
            l.set_ydata([u, u, up1])

        U.append(uk)

        arts[-1].set_text(timeform.format(frame*vmax/L) + unitform)
        return arts

    frames = t0 + dt*np.arange(0, M)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)

    plt.show()

if __name__=='__main__':
    animate(2**10)
