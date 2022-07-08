import numpy as np
import scipy.linalg as slin
import scipy.sparse.linalg as splin

from linalg.krylov import gmres
from linalg.multigrid import R, P


def compute_eta(etakm1: float, rkm1: float, rk: float, eta_max=0.1, gamma=0.5, epsilon=1e-9):
    """"
    Computes the tolerance for the next gmres step.
    """
    a = gamma*(rk/rkm1)**2
    b = gamma*etakm1**2

    if b < 0.1:
        c = min(eta_max, a)
    else:
        c = min(eta_max, max(a, b))
    
    return min(eta_max, max(c, epsilon/(2*rk)))


def NK(F, J, u0, eta_max=0.1, max_it=10, M=None):
    """
    Solves the nonlinear system of equations F with jacobian J using a Newton-Krylov solver
    """
    rtol = 1e-9

    if M is None:
        M = lambda _, __: lambda x: x

    u = u0.copy()

    r0 = slin.norm(F(u0))
    rkm1 = np.nan
    rk = r0
    r = rk

    eta = eta_max

    etas = []
    residuals = [r0]
    nits = []
    sols = [u0]

    k = 0
    while r/r0 > rtol and k < max_it:
        jk = J(u)
        fk = F(u)
        mk_fun = M(jk, u.size[0])

        s, i, r = gmres(splin.aslinearoperator(jk), -fk, mk_fun, tol=eta)
        u += s

        rkm1 = rk
        rk = slin.norm(F(u))
        eta = compute_eta(eta, rkm1, rk)

        residuals.append(r/r0)
        etas.append(eta)
        nits.append(i)
        sols.append(u)
        k += 1

    return residuals, sols, etas, nits


def JFNK(F, u0, eta_max=0.1, max_it=10, M=None):
    """
    Solves the nonlinear system of equations F using a jacobian free Newton-Krylov solver
    """
    n = u0.shape[0]
    tol = 1e-9
    mrs = 1e-7

    if M is None:
        M = lambda _, s: splin.LinearOperator((s, s), lambda x: x)

    u = u0.copy()

    r0 = slin.norm(F(u0))
    rk = r0
    rkm1 = r0

    eta = eta_max

    # Creates a function which approximates multiplication of the jacobian
    def approx_J(Fy, y):
        S = y.shape[0]

        def derivative(q):
            norm_q = slin.norm(q)
            j_eps = mrs / norm_q if norm_q != 0 else 1
            return (F(y + j_eps * q.reshape((S, ))) - Fy) / j_eps

        # TODO: Move this responsibility to multigrid. Newton should not know about its preconditioner
        def j_wrapper(k):
            if k == S:
                return splin.LinearOperator((S, S), derivative)
            else:
                return R(2*k) * j_wrapper(2*k) * P(k)
        return j_wrapper

    nits = [0]
    residuals = [r0]
    j = 0
    while rk/r0 > tol and j < max_it:
        fk = F(u)
        jk = approx_J(fk, u)
        mk_fun = M(jk, n)

        s, i, r = gmres(jk(n), -fk, tol=eta, k_max=30, m_right=mk_fun)
        u += s

        rkm1 = rk
        rk = slin.norm(F(u))
        eta = compute_eta(eta, rkm1, rk)

        nits.append(i)
        residuals.append(r)
        print("Newton({}), nGMRES({}), q = {}".format(j, i, int(np.log10(rk/r0))))

        j += 1

    return u
