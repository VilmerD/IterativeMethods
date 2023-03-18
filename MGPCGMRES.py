"""
    Illustrate how multigrid works within GMRES
"""
# Numpy and scipy stuff
import numpy as np
import scipy.sparse.linalg as spsl
import scipy.linalg as spl

# Problems
import project.problems as problems
import matrices.matricies as mats

# Multigrid
from linalg.smoothers import JacobiSmoother
from linalg.krylov import gmres
import newton.precondition as pc
import linalg.multigrid as mg

# Plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation

# Misc
from time import time
from primefac import primefac

def MG2():
    """"
        Solve 1d poisson using multigrid
    """
    # Define grid, n is number of interior points
    n, L = 511, 1
    xx = np.linspace(0, L, n+2)
    xxinner = xx[1:-1]

    # Define initial error
    v0 = np.exp(-10*(xxinner - 0.50*L)**2)*(np.sin(23*np.pi*xxinner) + np.sin(7*np.pi*xxinner) + 1)

    # Define problem matrix
    A = mats.poisson1d(n)

    # Compute residual
    f0 = np.zeros_like(xxinner)
    r0 = f0 - A.dot(v0)
    e0 = np.zeros_like(xxinner)

    # Define preconditioner
    height, pre, post, gamma = 5, 4, 4, 1
    smoother = JacobiSmoother(w=2/3)
    grid_interface = mg.DefaultInterface1D()

    # Make coarse grid representaions
    XXI = [xxinner]
    for k in range(1, height):
        XXI.append(grid_interface.restrict(XXI[k-1]))
    
    # Setup plot
    fig, ax = plt.subplots(1, 1)

    # Cycle using mg
    ncyc = 1
    n0 = np.linalg.norm(r0)
    vk = v0
    rk = r0
    ek = e0.copy()
    print("Relative norm of residual at start: {}".format(n0))
    for k in range(0, height):
        # Pre-smooth
        for _ in range(0, pre): ek = smoother.smooth(A, ek, rk)

        # Coarse-grid correction
        rk_2h = grid_interface.restrict(rk - A*ek)

        # Post-smooth
        for _ in range(0, post): ek = smoother.smooth(A, ek, rk)

        # Show results
        rk = f0 - A.dot(vk)
        ek = e0.copy()
        nk = np.linalg.norm(rk)/n0
        levelk = mgs.grid
        for k in range(0, height):
            ax.plot(XXI[k], levelk.v, label="e_{}".format(k))
            levelk = levelk.next_level


    #ax.plot(xxinner, vk, label='Error')
    #ax.plot(xxinner, f0, label='RHS')
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    plt.legend()
    plt.show()

def MG():
    """"
        Solve 1d poisson using multigrid
    """
    # Define grid, n is number of interior points
    n, L = 2047, 1
    xx = np.linspace(0, L, n+2)
    xxinner = xx[1:-1]

    # Define rhs
    #f0 = np.exp(-1000*(xxinner - 0.66*L)**2) - 2*np.cos(5*np.pi*(xxinner-0.33*L))*np.exp(-100*(xxinner - 0.33)**2)
    f0 = 10.0*(xxinner < 0.20*L) - 10.0*(xxinner > 0.80*L) - 5.0*(xxinner < 0.40*L) + 5.0*(xxinner > 0.60*L) + 10*np.exp(-10000*(xxinner - 0.50*L)**2)
    f0 = 100*np.exp(-10*(xxinner - 0.50*L)**2)*(np.sin(47*np.pi*xxinner) * (47*np.pi)**2 + -10)
    #f0 = -2*np.sin(3*np.pi*xxinner)**4

    # Define problem matrix
    A = mats.poisson1d(n)
    v0 = np.zeros_like(xxinner)
    r0 = f0 - A.dot(v0)
    e0 = np.zeros_like(xxinner)

    # Define preconditioner
    height, pre, post, gamma = 10, 2, 3, 1
    smoother = JacobiSmoother(w=2/3)
    gridinf = mg.DefaultInterface1D()
    mgs = mg.Multigrid(f0, gridinf, height, A=A)
    mgs.smoother = smoother
    mgs.pre = pre
    mgs.post = post
    mgs.gamma = gamma

    # Cycle using mg
    ncyc = 7
    n0 = np.linalg.norm(r0)
    vk = v0
    rk = r0
    ek = e0.copy()
    print("Relative norm of residual at start: {}".format(n0))
    for k in range(0, ncyc):
        vk += mgs.cycle(rk, ek)
        rk = f0 - A.dot(vk)
        ek = e0.copy()
        nk = np.linalg.norm(rk)/n0
        print("Relative norm of residual after {} cycle(s): {}".format(k, nk))  


    # Plot some results
    fig, ax = plt.subplots(1, 1)
    ax.plot(xxinner, vk, label='Approximation')
    #ax.plot(xxinner, f0, label='RHS')
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    plt.legend()
    plt.show()

def visMG():
    """"
        Solve 1d poisson using multigrid and visualize results
    """
    # Define grid, n is number of interior points
    n, L = 1023, 1
    xx = np.linspace(0, L, n+2)
    xxinner = xx[1:-1]

    # Define rhs
    f0 = np.exp(-1000*(xxinner - 0.66*L)**2) + np.exp(-1000*(xxinner - 0.33)**2)
    f0 = 1.0*(xxinner < 0.33*L) - 12.0*(xxinner > 0.66*L)
    #f0 = -2*np.sin(3*np.pi*xxinner)**4

    # Define problem matrix
    A = mats.poisson1d(n)
    v0 = np.zeros_like(xxinner)
    r0 = f0 - A.dot(v0)
    e0 = np.zeros_like(xxinner)

    # Define Solver
    height, pre, post, gamma = 8, 2, 3, 1
    smoother = JacobiSmoother(w=2/3)
    gridinf = mg.DefaultInterface1D()
    mgs = mg.Multigrid(A, f0, gridinf, height)
    mgs.smoother = smoother
    mgs.pre = pre
    mgs.post = post
    mgs.gamma = gamma

    # Make coarse grid representaions
    XXI = [xxinner]
    for k in range(1, height):
        XXI.append(gridinf.restrict(XXI[k-1]))

    # Solve problem on coarse grids
    FF = [f0]
    UU = [spsl.spsolve(A, f0)]
    Aop = mg.ScalableLinearOperator(A, gridinf)
    for k in range(1, height):
        fk = gridinf.restrict(FF[k-1])
        nk = fk.shape[0]
        Ak = Aop.to_dense(nk)
        FF.append(fk)
        UU.append(np.linalg.solve(Ak, fk))

    # Make figure
    fig, ax = plt.subplots()
    ax.set_ylabel("y")
    ax.set_xlabel("x")

    # Cycle using mg
    ncyc = 5
    n0 = np.linalg.norm(r0)
    vk = v0
    rk = r0
    ek = e0.copy()
    print("Relative norm of residual at start: {}".format(n0/n0))
    for k in range(0, ncyc):
        # Get approximation of error
        ekapp = mgs.cycle(rk, ek)
        vkp1 = vk + ekapp

        # Correct current approximation
        rkp1 = f0 - A.dot(vkp1)
        ekp1 = e0.copy()
        nkp1 = np.linalg.norm(rkp1)/n0

        # Plot results
        fig, ax = plt.subplots()
        ax.set_ylabel("y")
        ax.set_xlabel("x")
        print("Relative norm of residual after {} cycle(s): {}".format(k, nkp1))  
        levelk = mgs.grid
        for k in range(0, height):
            ax.plot(XXI[k], (UU[k] - vk) - levelk.v, label='ek')
            levelk = levelk.next_level
        plt.show()

        vk = vkp1
        rk = rkp1
        ek = ekp1

    # Plot some results
    fig, ax = plt.subplots(1, 1)
    ax.plot(xxinner, UU[0] - vk, label='Final error')
    #ax.plot(xxinner, f0, label='RHS')
    plt.legend()
    plt.show()

def MGPCGMRES():
    """"
        Solve 1d poisson using multigrid-preconditioned gmres. Homogenous dirichlet bcs
    """
    # Define grid, n is number of interior points
    n, L = 1023, 1
    xx = np.linspace(0, L, n+2)
    xxinner = xx[1:-1]


    # Define rhs
    f = np.exp(-100*(xxinner - 0.30*L)**2)

    # Define problem matrix
    A = mats.poisson1d(n)
    v0 = np.zeros_like(xxinner)
    f0 = f.copy()

    # Define preconditioner
    height, pre, post, gamma = 6, 3, 3, 1
    smoother = JacobiSmoother(w=2/3)
    gridinf = mg.DefaultInterface1D()
    mgpre = pc.MultigridPreconditioner((n, n), gridinf, height, smoother, pre, post, gamma=gamma)
    #mgpre = pc.Preconditioner()
    mgpremat = mgpre.make(A)

    # Solve problem
    x, ninner, res = gmres(A, f, m_right = mgpremat)

    print("Number of gmres {}, final residual {}".format(ninner, res))

if __name__ == '__main__':
    MG2()