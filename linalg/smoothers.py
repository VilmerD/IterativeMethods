import scipy.sparse as sp
from scipy.sparse import extract
from scipy.sparse import linalg as splin

class Smoother():
    def smooth(self, A, x, b): return x

class RK2Smoother(Smoother):
    def __init__(self, a1, hfun):
        self.a1 = a1
        self.hfun = hfun

    def smooth(self, A, x, b):
        a1, h = self.a1, self.hfun(A.shape[0])
        return x + h*(A._matvec(x + a1*h*(A._matvec(x) - b)) - b)

class GaussSeidelSmoother(Smoother):
    def smooth(self, a, x, b):
        dl = extract.tril(a, 0)                             # lower plus diagonal matrix
        u = sp.csr_matrix(extract.triu(a, 1))               # upper matrix
        return splin.spsolve_triangular(dl, u.dot(x) + b)

class JacobiSmoother(Smoother):
    def __init__(self, w=2/3):
        self.w = w

    def smooth(self, a, x, b):
        d = a.diagonal()[0]
        return x - self.w * (a.dot(x) - b) / d
