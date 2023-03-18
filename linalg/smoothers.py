import scipy.sparse as sp
from scipy.sparse import extract
from scipy.sparse import linalg as splin

class Smoother():
    def smooth(self, A, x, b): return x

class RK2Smoother(Smoother):
    def __init__(self, L, nu, c=0.99, a1=0.33):
        self.a1 = a1
        self.hfun = lambda N: c*(L/N)/nu

    def smooth(self, A, x, b):
        h = self.hfun(A.shape[0])
        return x + h*(b-A._matvec(x + self.a1*h*(b-A._matvec(x))))

class GaussSeidelSmoother(Smoother):
    def smooth(self, a, x, b):
        dl = extract.tril(a, 0)                             # lower plus diagonal matrix
        u = sp.csr_matrix(extract.triu(a, 1))               # upper matrix
        return splin.spsolve_triangular(dl, u.dot(x) + b)

class JacobiSmoother(Smoother):
    def __init__(self, w=2/3):
        self.w = w

    def smooth(self, a, x, b):
        A = a.to_dense(x.shape[0])
        d = A.diagonal()[0]
        return x - self.w * (a._matvec(x) - b) / d
