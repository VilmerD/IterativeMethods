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

def MG3():
    """"
        Solve 1d poisson using multigrid
    """
    # Define grid, n is number of interior points
    n, L = 2**7-1, 1
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

    # Make coarse grid representaions
    XXI = [xxinner]
    AA = [spsl.aslinearoperator(A)]
    Ik, Rk = mg.createDefaultInterpolator(n)
    II = [Ik]
    RR = [Rk]
    for k in range(1, height):
        # Create interpolator and restrictor functions
        nkp1 = II[k-1].shape[1]
        Ikp1, Rkp1 = mg.createDefaultInterpolator(nkp1)
        II.append(Ikp1)
        RR.append(Rkp1)
        # Compute coarse-grid representation
        XXI.append(RR[k-1]*XXI[k-1])
        # Compute coarse-grid operator
        Ak = spsl.LinearOperator((nkp1, nkp1), lambda x: RR[k-1].matvec(AA[k-1].matvec(II[k-1].matvec(x))))
        AA.append(Ak)
    
    # Setup plot
    fig, ax = plt.subplots(1, 1)

    # Cycle using mg
    def cycle(Akh, ekh, rkh):
        # Plot error before presmoothing

        # Pre-smooth
        for _ in range(0, pre): ek = smoother.smooth(A, ekh, rkh)

        # Plot error after presmoothing

        # Coarse-grid correction
        rk2h = grid_interface.restrict(rkh - A*ekh)
        ek2h = cycle(rk2h)
        ekh += grid_interface.prolong(ek2h)

        # Plot error after coarse-grid correction

        # Post-smooth
        for _ in range(0, post): ek = smoother.smooth(A, ek, rk)

        # Plot error after post-smoothing

        # Return error estimation
        return ekh
    

    ax.set_ylabel("y")
    ax.set_xlabel("x")
    plt.legend()
    plt.show()

def MG2():
    """"
        Solve 1d poisson using multigrid
    """
    # Define grid, n is number of interior points
    n, L = 2**10-1, 1
    xx = np.linspace(0, L, n+2)
    xxinner = xx[1:-1]

    # Define problem matrix
    A = mats.poisson1d(n)

    # Define initial error and residual
    loadcase = 0
    if loadcase == 0:
        u = np.exp(-20*(xxinner - 0.50*L)**2)*(0 + \
            +0.40*np.sin(567*np.pi*xxinner) + \
            +0.20*np.sin(359*np.pi*xxinner) + \
            -0.30*np.sin(231*np.pi*xxinner) + \
            -0.30*np.cos(97*np.pi*xxinner) + \
            +0.60*np.sin(51*np.pi*xxinner) + \
            +1.00*np.cos(27*np.pi*xxinner) + \
            +0.70*np.sin(13*np.pi*xxinner) + \
            -1.00*np.cos(7*np.pi*xxinner))
    elif loadcase == 1:
        u = (3*(xxinner <= 0.33) - 3*(xxinner < 0.10)) - (3*(xxinner > 0.66) - 3*(xxinner >= 0.90))
    elif loadcase == 2:
        u = 3*np.exp(-200*(xxinner - 0.25*L)**2)*(np.sin(np.pi*xxinner*567) + np.sin(np.pi*xxinner*359)) - \
            3*np.exp(-200*(xxinner - 0.75*L)**2)*(1*np.sin(np.pi*(xxinner - 0.75*L)*27) - 1*np.sin(np.pi*(xxinner - 0.75*L)*3))
    
    v0 = np.zeros_like(xxinner)
    f0 = A*u

    # Define multigrid parameters
    height, pre, post, gamma = 6, 5, 5, 1
    smoother = JacobiSmoother(w=2/3)
    grid_inter = mg.DefaultInterface1D()
    A = mg.ScalableLinearOperator(spsl.aslinearoperator(A), grid_inter)

    # Make coarse grid representaions
    XXI = [xxinner]
    Ik, Rk = mg.createDefaultInterpolator(n)
    II = [Ik]
    RR = [Rk]
    for k in range(1, height):
        # Compute coarse-grid representation
        XXI.append(grid_inter.restrict(XXI[k-1]))

    # Setup plot
    fig, axs = plt.subplots(height, 4)
    axs[0, 0].title.set_text("Before pre-smoothing")
    axs[0, 1].title.set_text("After pre-smoothing")
    axs[0, 2].title.set_text("After coarse-grid correction")
    axs[0, 3].title.set_text("After post-smoothing")

    axs[0, 0].set_ylabel("Level 0 [Finest]")
    for k in range(1, height-1):
        axs[k, 0].set_ylabel("Level {}".format(k))
    axs[height-1, 0].set_ylabel("Level {} [Coarsest]".format(height-1))

    for k in range(0, height):
        for l in range(0, 4):
            axs[k, l].set_xlim([0, L])
            axs[k, l].set_ylim([-4, 4])
            axs[k, l].set_xticks([0, 0.50, 1])
            axs[k, l].set_yticks([-4, 0, 4])
            nk = XXI[k].shape[0]
            lk = 4*L / (nk + 1)
            #axs[k, l].plot([L-lk, L], [1.5, 1.5], color='black')

    # Cycle using mg
    level = 0
    def cycle(vkh, fkh, level):
        if level != height - 1:
            # Compute solution at level
            ukh = np.linalg.solve(A.to_dense(fkh.shape[0]), fkh)

            # Plot before pre-smoothing
            axs[level, 0].plot(XXI[level], vkh, color='green')
            axs[level, 0].plot(XXI[level], ukh - vkh, color='red')
            #axs[level, 0].plot(XXI[level], vkh, color='green')

            # Pre-smooth
            for _ in range(0, pre): vkh = smoother.smooth(A, vkh, fkh)

            # Plot error after presmoothing
            axs[level, 1].plot(XXI[level], vkh, color='green')
            axs[level, 1].plot(XXI[level], ukh - vkh, color='red')
            #axs[level, 1].plot(XXI[level], vkh, color='green')

            # Coarse-grid correction
            for _ in range(0, gamma): 
                fk2h = grid_inter.restrict(fkh - A._matvec(vkh))
                vk2h = cycle(np.zeros_like(fk2h), fk2h, level+1)
                vkh += grid_inter.prolong(vk2h)

            # Plot error after coarse-grid correction
            axs[level, 2].plot(XXI[level], vkh, color='green')
            axs[level, 2].plot(XXI[level], ukh - vkh, color='red')
            #axs[level, 2].plot(XXI[level], vkh, color='green')

            # Post-smooth
            for _ in range(0, post): vkh = smoother.smooth(A, vkh, fkh)
            
            # Plot error after post-smoothing
            axs[level, 3].plot(XXI[level], vkh, color='green')
            axs[level, 3].plot(XXI[level], ukh - vkh, color='red')
            #axs[level, 3].plot(XXI[level], vkh, color='green')
        else:
            # Solve problem exactly on coarsest level
            vkh = np.linalg.solve(A.to_dense(fkh.shape[0]), fkh)

            # Plot error estimation on coarsest level
            axs[level, 2].plot(XXI[level], vkh, color='green')
            axs[level, 2].plot(XXI[level], np.zeros_like(vkh), color='red')
            #axs[level, 2].plot(XXI[level], vkh, color='green')

        # Return error estimation
        return vkh
    
    cycle(v0, f0, 0)
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