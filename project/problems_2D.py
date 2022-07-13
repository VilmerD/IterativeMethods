import numpy as np
from problems import zero_periodic

def F2(u, dt, dx, uold):
    n = int(len(u) ** 0.5)
    usqr = u ** 2
    uoldsqr = uold ** 2
    return u - uold + dt / (4 * dx) * (2 * usqr - np.roll(usqr, -n) - np.roll(usqr, -1)
                                       + 2 * uoldsqr - np.roll(uoldsqr, -n) - np.roll(uoldsqr, -1))


@zero_periodic
def ua_2D(x):
    n = x.shape[0]
    xx, yy = np.meshgrid(x, x)
    mat = 2 + 2*np.sin(np.pi / 8 * (xx - yy - 4)) * np.sin(np.pi / 8 * xx)
    return mat.reshape((n ** 2,))