import numpy as np
import scipy.sparse.linalg as splin


def v_cycle(a, v0, f0, smoother, gamma=1, l0=3, pre=1, post=1):
    # Initialize grid
    grid = Grid(v0, f0, l0)

    def v_cycle_recursive(level):
        v, f = Grid.get_level(level)
        if level > 0:
            vm1, fm1 = Grid.get_level(level - 1)

            # Presmooth
            for _ in range(0, pre):
                v = smoother(a, v, f)
            fm1 = R.dot(a.dot(v) - f)
    
            # Dive deeper
            for _ in range(0, gamma):
                v_cycle_recursive(level - 1)

            # Correction
            v = v - P.dot(vm1)

            # Postsmooth
            for _ in range(0, post):
                v = smoother(a, v, f)
        else:
            v = splin.spsolve(a.dot(np.eye(grid.bottom)), f)
    
    # Start cycle at top level
    v_cycle_recursive(grid.height-1)

    # Return top level solution
    return grid.get_level(grid.height-1)[0]


def FAS(A, v0, f, smoother, gamma=3, level0=5, pre=30):
    grid = Grid(v0, f, level0)
    epsilon = 0.9

    def FAS_recursive(level):
        current_level = grid.levels[level]
        next_level = grid.levels[level - 1]

        for k in range(0, pre):
            current_level.v = smoother(A, current_level.v, current_level.f)

        if level > 0:
            n = current_level.v.shape[0]
            u_tilde = R(n) * current_level.v
            next_level.f = A(int(n / 2)) * u_tilde + epsilon * R(n) * (current_level.f - A(n) * current_level.v)
            for k in range(0, gamma):
                FAS_recursive(level - 1)
            current_level.v += P(int(n / 2)) * (next_level.v - u_tilde) / epsilon

    FAS_recursive(grid.height)
    return grid.levels[-1].v


def R(n):
    return splin.LinearOperator((int(n / 2), n), agg_res)


def P(n):
    return splin.LinearOperator((2 * n, n), agg_pro)


def aggres2d(v):
    n = int(v.shape[0] ** 0.5)
    vmat = v.reshape((n, n))
    mat = 0.25 * (vmat[0::2, 0::2] + vmat[1::2, 1::2] + vmat[0::2, 1::2] + vmat[1::2, 0::2])
    return mat.reshape((int(n ** 2 / 4), ))


def aggpro2d(v):
    n = int(v.shape[0] ** 0.5)
    vs = v.reshape((n, n))
    vnew = np.zeros((n * 2, n * 2))
    vnew[0::2, 0::2] = vs
    vnew[1::2, 1::2] = vs
    vnew[1::2, 0::2] = vs
    vnew[0::2, 1::2] = vs
    return vnew.reshape((4 * n ** 2, ))


def default_res(v):
    return (v[0:-2:2] + 2 * v[1:-1:2] + v[2::2]) / 4


def default_pro(v):
    u = np.zeros(len(v) * 2 + 1)
    u[1:-1:2] = v
    return (2 * u + np.roll(u, 1) + np.roll(u, -1)) / 2


def agg_res(v):
    return (v[:-1:2] + v[1::2]) / 2


def agg_pro(v):
    return np.repeat(v, 2)


class Restrictor:
    pass


class Prolonger:
    pass


class Grid:
    levels: list
    ndims: int

    def __init__(self, v0, f0, bottom, ndims=1):
        self.ndims = ndims
        top = int(np.log(v0.shape[0])/np.log(2)/ndims)
        self.levels = [Grid.Level(2**(ndims*n)) for n in range(bottom, top)]
        self.levels.append(Grid.Level(top, v0=v0, f0=f0))

    def _get_top(self):
        return self.levels[0].size

    def _get_bottom(self):
        return self.levels[-1].size

    def _get_height(self):
        return len(self.levels)

    def get_level(self, level):
        v = self.levels[level].v
        f = self.levels[level].f
        return (v, f)

    top = property(_get_top)
    bottom = property(_get_bottom)
    height = property(_get_height)

    class Level:
        def __init__(self, n, v0=None, f0=None):
            self.v = np.zeros(n) if v0 is None else v0
            self.f = np.zeros(n) if f0 is None else f0

        def _get_size(self):
            return self.v.shape[0]

        size = property(_get_size)


v0 = np.zeros(128)
f0 = v0.copy()
g = Grid(v0, f0, 5)
for l in g.levels:
    print(l.v.shape)
print(g.top)
print(g.bottom)
print(g.height)

v = np.arange(0, 8)
vm1 = agg_res(v)
print(v)
print(vm1)
v = agg_pro(vm1)
print(v)
print(vm1)