import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator

from linalg.precond import Preconditioner
from linalg.krylov import gmres


def compute_eta(etakm1: float, rk: float, rkm1: float, eta_max=0.1, gamma=0.5, epsilon=1e-9):
    """"
    Computes the tolerance for the next step
    """
    a = gamma*(rk/rkm1)**2
    b = gamma*etakm1**2

    if b < 0.1:
        c = min(eta_max, a)
    else:
        c = min(eta_max, max(a, b))
    
    return min(eta_max, max(c, epsilon/(2*rk)))


def NK(F, J, u0: np.array, eta_max=0.1, max_it=10, M=None):
    """
    Solves the nonlinear system of equations F with jacobian J using a Newton-Krylov solver
    """
    rtol = 1e-9

    # Initialize preconditioner
    if M is None:
        M = Preconditioner(J)

    # Compute initial rhs, residual and target tolerance 
    f0 = F(u0)
    r0 = sla.norm(f0)
    eta = eta_max

    # Initialize lists (solutions, residuals, inner iterations, etas)
    lists = [u0], [r0], [0], [eta]

    # Initialize loop
    k, uk, fk, rk = 0, u0.copy(), f0, r0
    while rk/r0 > rtol and k < max_it:
        # Linearize around current solution
        jk = J(uk, fk)
        mk = M.make(jk)

        # Solve linearized system
        dk, ik, _ = gmres(jk, -fk, tol=eta, m_right=mk)
        ukp1 = uk + dk
        fkp1 = F(ukp1)
        rkp1 = sla.norm(fkp1)

        # Update quantities
        uk, fk, rk, eta = ukp1, fkp1, rkp1, compute_eta(eta, rkp1, rk)
        
        for l, i in zip(lists, (uk, rk, ik, eta)): l.append(i)
        k += 1

    return lists

def JFNK(F, u0: np.array, eta_max=0.1, max_it=10, M=None):
    """
    Solves nonlinear system of equations F using a Jacobian-Free Newton-Krylov solver
    """
    # Define Jacobian approximation
    sqreps = np.sqrt(np.finfo(float).eps)
    def Jfun(uk, fk, d):
        dn = sla.norm(d)        
        h = 1 if dn == 0 else sqreps/dn
        return (F(uk + h * d) - fk)/h
    
    # J is a function which returns a linear operator given 
    Jshape = (u0.shape[0], u0.shape[0])
    matvec = lambda uk, fk: lambda x: Jfun(uk, fk, x)
    J = lambda uk, fk: LinearOperator(Jshape, matvec(uk, fk))

    # Solve using NK
    return NK(F, J, u0, eta_max=eta_max, max_it=max_it, M=M)
