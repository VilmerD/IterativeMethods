from re import X
import numpy as np


# Initial conditions
def zero_periodic(func):
    def eval_wrapper(x, *args):
        fx = func(x, *args)
        fxs = np.roll(fx, -1)
        return (fx + fxs) / 2
    return eval_wrapper

# Conservation law discretized
def conservation_law_periodic(flux):
    def wrapper(u, uold, dt, dx, *args):
        return u - uold + dt/dx * (flux(u, *args) - flux(np.roll(u, -1), *args))
    return wrapper

# Flux
@conservation_law_periodic
def flux(u):
    """ Computes the flux of materia in 1D continuum """
    return u**2/2

@zero_periodic
def ua(x):
    return 2 + np.sin(np.pi * x)

