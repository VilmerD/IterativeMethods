import numpy as np


# Initial conditions
def evalu0(func):
    def eval_wrapper(x):
        return (func(x) + func(np.roll(x, -1))) / 2

    return eval_wrapper


# Fluxes
def F(u, dt, dx, uold):
    return u - uold + dt / (2 * dx) * (u ** 2 - np.roll(u, -1) ** 2)


@evalu0
def func_u0(x):
    return 2 + np.sin(np.pi * x)


def F2(u, dt, dx, uold):
    n = int(len(u) ** 0.5)
    usqr = u ** 2
    uoldsqr = uold ** 2
    return u - uold + dt / (4 * dx) * (2 * usqr - np.roll(usqr, -n) - np.roll(usqr, -1)
                                       + 2 * uoldsqr - np.roll(uoldsqr, -n) - np.roll(uoldsqr, -1))


@evalu0
def func_u02(x):
    n = x.shape[0]
    xx, yy = np.meshgrid(x, x)
    mat = 2 + 2*np.sin(np.pi / 8 * (xx - yy - 4)) * np.sin(np.pi / 8 * xx)
    return mat.reshape((n ** 2,))
