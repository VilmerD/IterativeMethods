import numpy as np
from scipy.sparse.linalg import LinearOperator
import scipy.linalg as scla

from linalg.smoothers import RK2Smoother, Smoother

class GridInterface():
    def restrict(self, v) -> np.array: pass

    def prolong(self, v) -> np.array: pass

class AggregateInterface1D(GridInterface):
    def restrict(self, v):
        if v.shape[0] % 2 == 0:
            return (v[:-1:2] + v[1::2]) / 2
        else: 
            raise ValueError()

    def prolong(self, v):
        return np.repeat(v, 2)

class AggregateInterface2D(GridInterface):
    def restrict(v):
        n = int(v.shape[0] ** 0.5)
        vmat = v.reshape((n, n))
        mat = 0.25 * (vmat[0::2, 0::2] + vmat[1::2, 1::2] + vmat[0::2, 1::2] + vmat[1::2, 0::2])
        return mat.reshape((int(n ** 2 / 4), ))

    def prolong(v):
        n = int(v.shape[0] ** 0.5)
        vs = v.reshape((n, n))
        vnew = np.zeros((n * 2, n * 2))
        vnew[0::2, 0::2] = vs
        vnew[1::2, 1::2] = vs
        vnew[1::2, 0::2] = vs
        vnew[0::2, 1::2] = vs
        return vnew.reshape((4 * n ** 2, ))

class DefaultInterface1D(GridInterface):
    def restrict(self, v):
        if (v.shape[0] - 1) % 2 == 0:
            return (v[0:-2:2] + 2 * v[1:-1:2] + v[2::2]) / 4
        else:
            raise ValueError()

    def prolong(self, v):
        u = np.zeros(len(v) * 2 + 1)
        u[1:-1:2] = v
        return (2 * u + np.roll(u, 1) + np.roll(u, -1)) / 2


class ScalableLinearOperator(LinearOperator):
    def __init__(self, linop: LinearOperator, grid_int: GridInterface):
        self.shape = linop.shape
        self.linop = linop
        self.grid_interface = grid_int

        # Initialize possible shapes
        v = np.zeros(self.shape[0])
        self.shapes = [v.shape]
        has_next = True
        while has_next:
            try:
                v = grid_int.restrict(v)
                self.shapes.append(v.shape)
            except ValueError:
                has_next = False

    def _matvec(self, v):
        if v.shape in self.shapes:
            if v.shape == self.shapes[0]:
                return self.linop.dot(v)
            else:
                vm1 = self._matvec(self.grid_interface.prolong(v))
                return self.grid_interface.restrict(vm1)
        else:
            raise ValueError


    def _matmat(self, V):
        if (V.shape[0], ) in self.shapes:
            AV = []
            for r in V:
                AV.append(self._matvec(r))
            return np.array(AV)
        else:
            raise ValueError

    def to_dense(self, n):
        return self._matmat(np.eye(n))

class Multigrid:
    def __init__(self, v0: np.array, f0: np.array, grdint: GridInterface, height: int):
        self.v, self.f = v0, f0

        self.prolong = grdint.prolong
        self.restrict = grdint.restrict
        if height > 0:
            vm1 = grdint.restrict(v0)
            fm1 = grdint.restrict(f0)

            self.next_level = Multigrid(vm1, fm1, grdint, height-1)
        else:
            self.next_level = None


    def reset_v(self):
        """ Resets the grid levels """
        self.v = np.zeros_like(self.v)
        if self.hasNext():
            self.next_level.reset_v()

    def hasNext(self):
        return self.next_level is not None


def cycle(A: ScalableLinearOperator, top_floor: Multigrid, smoother: Smoother, pre=3, post=3, gamma=1):
    """
    Solves problem

    Arguments
        smoother:       smoother
        pre:            number of presmoothing cycles
        post:           number of postsmoothing cycles
        gamma:          

    Returns
        x:              solution

    """
    def cycle_recursive(floor: Multigrid):
        if floor.hasNext():
            # Presmooth
            for _ in range(0, pre): floor.v = smoother.smooth(A, floor.v, floor.f)
            
            # Move error to next level
            floor.next_level.f = floor.restrict(A._matvec(floor.v) - floor.f)
    
            # Coarse grid correction
            for _ in range(0, gamma): cycle_recursive(floor.next_level)
            floor.v -= floor.next_level.prolong(floor.next_level.v)

            # Postsmooth
            for _ in range(0, post): floor.v = smoother.smooth(A, floor.v, floor.f)
        else:
            # Solve exactly
            floor.v = scla.solve(A.to_dense(floor.v.shape[0]), floor.f)

    if scla.norm(top_floor.f) != 0:
        # Start cycle at top level
        cycle_recursive(top_floor)

        # Return top level solution
        return top_floor.v
    else:
        return np.zeros_like(top_floor.f)

def test():
    # Initialize grid
    N, L = 128, 1
    dx = L/(N + 1)
    xx = np.linspace(0, 1, N+2)[1:-1]

    # Initialize rhs
    f = 4
    f0 = -(f*np.pi)**2*np.sin(xx*f*np.pi)
    v0 = np.zeros_like(f0)

    # Initialize second order discretization
    C = np.hstack((np.array([-2, 1]), np.zeros(N-2)))/dx**2
    A = scla.toeplitz(C)

    # Setup multigrid
    top = Multigrid(A, v0, f0, AggregateInterface1D(), 4)

    # Smoother
    smoother = RK2Smoother(0.33, lambda N: 1e-3/N)

    # Solve
    x = cycle(top, smoother, pre=10, post=10, gamma=3)

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(xx, x, 'g')
    ax.plot(xx, scla.solve(A, f0), 'k')
    ax.legend(('Approx', 'Solution'))
    plt.show()

if __name__=='__main__':
    test()