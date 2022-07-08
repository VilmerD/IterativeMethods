from matrices.nonlinear_matricies import *

from newton.newton import NK
from newton.preconditioners import *

from linalg.smoothers import *

import matplotlib.pyplot as plt

from time import time


def A(n):
    return splin.aslinearoperator(sp.diags((1, -2, 1), (-1, 0, 1), (n, n)))


n = 2 ** 8
x = interval(n)
u = np.sin(3 * np.pi * x) + 1 / 9 * np.sin(21 * np.pi * x)
u0 = u.copy()


def RK2(A, x, b):
    a1, c1 = 0.33, 0.99
    N = b.shape[0]
    h = c1 / (3 * 0.0052 * N)

    return x + h * A(N) * x + a1 * h ** 2 * A(N) * A(N) * b


for k in range(0, 10):
    u = RK2(A, u, np.zeros(u.shape))


fig, ax = plt.subplots()
plt.plot(x, u0, 'g', x, u, 'b')
plt.show()
