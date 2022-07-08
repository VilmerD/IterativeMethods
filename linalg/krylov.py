import numpy as np
import scipy.linalg as splin


def gmres(A, b, tol=1e-9, x0=None, k_max=None, m_right=None):
    """
    Solves the linear system of equations Ax = b using GMRES

    Arguments:
        A:          lhs (implements dot)
        b:          rhs (numpy array)
        tol:        relative tolerance (float)
        x0:         initial guess (numpy array)
        k_max:      maximum inner iterations (integer)
        m_right:    right preconditioner (implements dot)

    Returns:
        x:          solution (numpy array)
        j:          number of iterations (integer)
        gamma:      residual at last step (float)
    """
    # Initialize quantities
    if x0 is None:
        x0 = np.zeros_like(b)
    
    # Default is 40, otherwise make sure number of steps is not more than size
    if k_max is None:
        k_max = 40
    else:
        k_max = min(b.shape[0], k_max)

    # Set up preconditioner, default is identity matrix
    if m_right is None:
        m_right = np.eye(b.shape[0])
    
    r0 = b - A.dot(x0)
    gamma = [splin.norm(r0)]

    if gamma[0] < tol:
        x = x0
        return x, 1, gamma[0]

    v = [r0 / gamma[0]]
    h = np.zeros((k_max + 1, k_max))

    c = np.zeros(k_max)
    s = np.zeros(k_max)

    for j in range(0, k_max):
        # Expand subspace
        wj = A.dot(m_right.dot(v[j]))
        for i in range(0, j + 1):
            h[i, j] = v[i].dot(wj)
            wj -= h[i, j] * v[i]

        # Update H
        h[j + 1, j] = splin.norm(wj)
        for i in range(0, j):
            hij = c[i]*h[i, j] + s[i]*h[i+1, j]
            hip1j = -s[i]*h[i, j] + c[i]*h[i+1, j]
            h[i, j], h[i+1, j] = hij, hip1j

        # Perform givens rotation
        beta = np.sqrt(h[j, j] ** 2 + h[j + 1, j] ** 2)
        s[j] = h[j + 1, j] / beta
        c[j] = h[j, j] / beta
        h[j, j] = beta

        # Update gamma
        gamma.append(-s[j]*gamma[j])
        gamma[j] = c[j]*gamma[j]

        # Check for convergance
        if np.abs(gamma[j + 1]/gamma[0]) >= tol:
            v.append(wj / h[j + 1, j])
        else:
            break
    
    # Compute solution
    alpha = np.zeros((j + 1, 1))
    y = np.zeros_like(b)
    for i in range(j, -1, -1):
        alpha[i] = (gamma[i] - h[i, i + 1: j + 1].dot(alpha[i + 1:j + 1])) / h[i, i]
        y += alpha[i] * v[i]
        
    # Recast to x
    x = m_right.dot(y)
    return x, j+1, abs(gamma[j+1]/gamma[0])
