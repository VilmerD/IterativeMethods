import numpy as np
import scipy.sparse as scsp
import scipy.sparse.linalg as scspla

import multigrid as mg

class Preconditioner:
    def __init__(self, A):
        self.shape = A.shape
        self.P = scsp.eye(A.shape)

    def make(self, _) -> scsp.linalg.LinearOperator:
        return self.P

class MultigridPreconditioner(Preconditioner):
    def __init__(self, shape, grint, height, smoother, pre, post, gamma):
        self.shape = shape
        v0 = np.zeros(shape[0])
        self.grid = mg.Multigrid(v0, v0, grint, height)
        self.grint = grint

        self.smoother = smoother
        self.pre = pre
        self.post = post
        self.gamma = gamma

    def solve(self, x):
        self.grid.reset_v()
        self.grid.f = x
        return mg.cycle(self.A, self.grid, self.smoother, \
            pre=self.pre, post=self.post, gamma=self.gamma)

    def make(self, A):
        self.A = mg.ScalableLinearOperator(A, self.grint)
        return scspla.LinearOperator(self.shape, self.solve)


class ILUPreconditioner(Preconditioner):
    def __init__(self, shape):
        self.shape = shape

    def make(self, A):
        Ainv = scsp.linalg.s(A)
        return scsp.linalg.LinearOperator(Ainv.shape, Ainv.solve)