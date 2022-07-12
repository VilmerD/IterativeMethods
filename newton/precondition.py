import numpy as np
import scipy.sparse as scsp
import scipy.sparse.linalg as scspla

import linalg.multigrid as mg

class Preconditioner:
    def make(self, A):
        return scsp.eye(A.shape[0])

class MultigridPreconditioner(Preconditioner):
    def __init__(self, shape, grdif, height, smoother, pre, post, gamma):
        self.shape = shape
        v0 = np.zeros(shape[0])
        f0 = v0
        multigrid = mg.Multigrid(v0, f0, grdif, height)

        multigrid.smoother = smoother
        multigrid.pre = pre
        multigrid.post = post
        multigrid.gamma = gamma
        
        self.multigrid = multigrid

    def make(self, A):
        Ascl = mg.ScalableLinearOperator(A, self.multigrid.grdif)
        solve = lambda x: self.multigrid.cycle(Ascl, x)
        return scspla.LinearOperator(self.shape, solve)

class ILUPreconditioner(Preconditioner):
    def make(self, A):
        Ainv = scspla.spilu(A)
        return scsp.linalg.LinearOperator(Ainv.shape, Ainv.solve)