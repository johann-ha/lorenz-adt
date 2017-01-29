import numpy as np

class Integrand(object):
    pass

def integrate(model, dt):
    """ Explicit Euler Formula. """
    return model + model.t()*dt