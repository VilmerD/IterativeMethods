import numpy as np
from scipy.sparse import eye as speye
import scipy.linalg as scla


def gmres(A, b, tol=1e-9, x0=None, k_max=None, m_right=None, restart=False):
    """
    Solves the linear system of equations Ax = b using GMRES

    Arguments:
        A:          lhs
        b:          rhs
        tol:        relative tolerance
        x0:         initial guess
        k_max:      maximum inner iterations
        m_right:    right preconditioner

    Returns:
        x:          solution
        j:          number of iterations
        gamma:      residual at last step
    """
    # Initialize x
    if x0 is None:
        x0 = np.zeros_like(b)
    
    # Default is 40, otherwise make sure number of steps is not more than size
    if k_max is None:
        k_max = 40
    else:
        k_max = min(b.shape[0], k_max)

    # Set up preconditioner, default is identity matrix
    if m_right is None:
        m_right = speye(b.shape[0])
    
    # Initial residual
    r0 = b - A.dot(x0)
    gamma = [scla.norm(r0)]
    if gamma[0] < tol:
        x = x0
        return x, 1, gamma[0]

    # Allocate memory for krylov subspace and H
    v = [r0 / gamma[0]]
    h = np.zeros((k_max + 1, k_max))

    # Allcoate memory for givens rotation
    c = np.zeros(k_max)
    s = np.zeros(k_max)

    # Starts loop
    for j in range(0, k_max):
        # Do Arnoldi iteration: expand subspace and update H
        wj = A.dot(m_right.dot(v[j]))
        for i in range(0, j + 1):
            h[i, j] = v[i].dot(wj)
            wj -= h[i, j] * v[i]
        h[j + 1, j] = scla.norm(wj)

        # Perform givens rotation
        for i in range(0, j):
            hij     = +c[i]*h[i, j] +s[i]*h[i+1, j]
            hip1j   = -s[i]*h[i, j] +c[i]*h[i+1, j]
            h[i, j], h[i+1, j] = hij, hip1j
        beta = np.sqrt(h[j, j] ** 2 + h[j + 1, j] ** 2)
        s[j] = h[j + 1, j] / beta
        c[j] = h[j, j] / beta
        h[j, j] = beta

        # Update gamma
        gamma.append(-s[j]*gamma[j])
        gamma[j] = c[j]*gamma[j]

        # Check for convergance
        if abs(gamma[j + 1]/gamma[0]) < tol:
            break
        else:
            # Append next vector
            v.append(wj / h[j + 1, j])
    
    # Compute solution
    alpha = np.zeros((j + 1, 1))
    y = np.zeros_like(b)
    for i in range(j, -1, -1):
        alpha[i] = (gamma[i] - h[i, i + 1: j + 1].dot(alpha[i + 1:j + 1])) / h[i, i]
        y += alpha[i] * v[i]

    # Recast to x and compute final quantities
    x = m_right.dot(y)
    ninner, relres = j+1, abs(gamma[j+1]/gamma[0])

    # Restart if necessary
    if restart and relres > tol:
        x, ninner2, relres = gmres(A, b, tol=tol, x0=x, k_max=k_max, restart=False)
        return x, ninner+ninner2, relres
    else:
        return x, ninner, relres
