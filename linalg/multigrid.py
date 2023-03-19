import numpy as np
from scipy.sparse.linalg import LinearOperator
import scipy.linalg as scla

from linalg.smoothers import Smoother

def createDefaultInterpolator(n_current: int):
    """ 
        Creates interpolator matrix for default interpolation
    """
    n_coarser = int((n_current - 1)/2) 
    I = LinearOperator((n_current, n_coarser), lambda x: default_prolong(x))
    R = LinearOperator((n_coarser, n_current), lambda x: default_restrict(x))
    return I, R

def default_restrict(v):
    if (v.shape[0] - 1) % 2 == 0:
        return (v[0:-2:2] + 2 * v[1:-1:2] + v[2::2]) / 4
    else:
        raise ValueError()

def default_prolong(v):
    u = np.zeros(len(v) * 2 + 1)
    u[1:-1:2] = v
    return (2 * u + np.roll(u, 1) + np.roll(u, -1)) / 2

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
        self.shapes = [v.shape[0]]
        has_next = True
        while has_next:
            try:
                v = grid_int.restrict(v)
                self.shapes.append(v.shape[0])
            except ValueError:
                has_next = False

    def _matvec(self, v):
        if v.shape[0] in self.shapes:
            if v.shape[0] == self.shapes[0]:
                return self.linop.dot(v)
            else:
                vm1 = self._matvec(self.grid_interface.prolong(v))
                return self.grid_interface.restrict(vm1)
        else:
            raise ValueError

    def _matmat(self, V):
        if V.shape[0] in self.shapes:
            return np.hstack([self._matvec(c).reshape(-1, 1) for c in V.T])
        else:
            raise ValueError

    def to_dense(self, n):
        return self._matmat(np.eye(n))

class Multigrid:
    pre, post, gamma = 3, 3, 1
    smoother = Smoother()

    def __init__(self, f: np.array, grdif: GridInterface, height: int, A: np.array = None):
        self.grdif = grdif
        if A is not None: self.setOperator(A)
        v0 = np.zeros_like(f)
        f0 = f
        self.grid = Multigrid.Grid(v0, f0, grdif, height)

    def setOperator(self, A: any):
        if ~isinstance(A, ScalableLinearOperator):
            self.op = ScalableLinearOperator(A, self.grdif)
        else:
            self.op = A

    def hasNext(self): return self.next_level is not None

    def cycle(self, f0: np.array = None, v0: np.array = None):
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
        # Set operator

        # Reset the grid
        self.grid.set(f0=f0, v0=v0)

        def cycle_recursive(floor: Multigrid.Grid):
            if floor.hasNext():
                # Presmooth
                for _ in range(0, self.pre): floor.v = self.smoother.smooth(self.op, floor.v, floor.f)
                
                # Move error to next level
                floor.next_level.f = floor.grdif.restrict(floor.f - self.op._matvec(floor.v))
        
                # Coarse grid correction
                for _ in range(0, self.gamma): cycle_recursive(floor.next_level)
                floor.v += floor.next_level.grdif.prolong(floor.next_level.v)

                # Postsmooth
                for _ in range(0, self.post): floor.v = self.smoother.smooth(self.op, floor.v, floor.f)
            else:
                # Solve exactly
                floor.v = scla.solve(self.op.to_dense(floor.v.shape[0]), floor.f)

        if scla.norm(self.grid.f) != 0:
            # Start cycle at top level
            cycle_recursive(self.grid)

            # Return top level solution
            return self.grid.v
        else:
            # Zero vector is solution
            return np.zeros_like(self.grid.v)

    class Grid():
        def __init__(self, v0, f0, grdif, height):
            self.v, self.f = v0, f0
            self.next_level = None
            self.grdif = grdif
            if height > 1:
                vm1 = grdif.restrict(v0)
                fm1 = grdif.restrict(f0)
                self.next_level = Multigrid.Grid(vm1, fm1, grdif, height-1)
        
        def hasNext(self): return self.next_level is not None

        def set(self, f0=None, v0=None):
            """ Resets the grid levels """
            self.f = f0 if f0 is not None else np.zeros_like(self.f)
            self.v = v0 if v0 is not None else np.zeros_like(self.v)
            if self.hasNext(): self.next_level.set()
