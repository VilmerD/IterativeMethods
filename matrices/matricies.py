import numpy as np
import scipy.sparse as sp

def discrete_second(n, L=1):
    dx = L / (n + 1)
    return sp.csr_matrix(dx ** -2 * sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n)))


