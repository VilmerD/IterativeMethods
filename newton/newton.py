import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator

from newton.precondition import Preconditioner
from linalg.krylov import gmres


def compute_eta(rk: float, rkm1: float, etakm1: float, rtol: float, \
    eta_max=0.1, gamma=0.5):
    """"
    Computes the tolerance for the next step
    """
    # Start rough, and refine
    a = gamma*(rk/rkm1)**2
    b = min(eta_max, a)         # Ensure refine
    c = gamma*etakm1**2         

    if c <= 0.1:
        d = b
    else:
        d = min(eta_max, max(a, c))
    
    # Prevent oversolving
    eta = min(eta_max, max(d, (rtol/2)/rk))
    return eta


def NK(F, J, u0: np.array, rtol=1e-6, eta_max=0.1, max_it=10, M=None):
    """
    Solves the nonlinear system of equations F with jacobian J using a Newton-Krylov solver
    """
    # Initialize preconditioner
    if M is None:
        M = Preconditioner()

    # Compute initial rhs, residual and target tolerance 
    f0 = F(u0)
    r0 = sla.norm(f0)
    eta = eta_max

    # Initialize lists (solutions, residuals, inner iterations, etas)
    lists = [u0], [r0], [0], [eta]

    # Initialize loop
    k, uk, fk, rk = 0, u0.copy(), f0, r0
    print(f'Starting Newton with r0={r0}')
    while rk/r0 > rtol and k < max_it:
        # Linearize around current solution
        jk = J(uk, fk)
        mk = M.make(jk)

        # Solve linearized system
        dk, ik, _ = gmres(jk, -fk, tol=eta*rk, m_right=mk)
        ukp1 = uk + dk
        fkp1 = F(ukp1)
        rkp1 = sla.norm(fkp1)

        # Update quantities
        uk, fk, rk, eta = ukp1, fkp1, rkp1, compute_eta(rkp1, rk, eta, rtol)
        
        for l, i in zip(lists, (uk, rk, ik, eta)): l.append(i)
        k += 1

        # Update user
        print(f'\t Step accepted (rkp1/rk={eta}) GMRES iterations: {ik}')
    print(f'Newton converged (rk/r0={rtol})')
    print(f'\t total GMRES iterations: {sum(lists[2])}\n')
    return lists

def JFNK(F, u0: np.array, rtol=1e-6, eta_max=0.1, max_it=10, M=None):
    """
    Solves nonlinear system of equations F using a Jacobian-Free Newton-Krylov solver
    """
    # Define Jacobian approximation
    sqreps = np.sqrt(np.finfo(float).eps)
    def Jfun(uk, fk, d):
        dn = sla.norm(d)        
        h = 1 if dn == 0 else sqreps/dn
        return (F(uk + h * d) - fk)/h
    
    Jshape = (u0.shape[0], u0.shape[0])
    matvec = lambda uk, fk: lambda x: Jfun(uk, fk, x)
    J = lambda uk, fk: LinearOperator(Jshape, matvec(uk, fk))

    # Solve using NK
    return NK(F, J, u0, rtol=rtol, eta_max=eta_max, max_it=max_it, M=M)
