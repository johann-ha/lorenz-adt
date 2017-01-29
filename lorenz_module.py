import numpy as np
from integrand_module import Integrand
import integrand_module as integrand

class Lorenz(Integrand):
    def __init__(self, initial_state, s=0, r=0, b=0):
        self.state = initial_state
        self.sigma = s
        self.rho = r
        self.beta = b
        self.dimension = initial_state.shape

    def output(self):
        coordinates = self.state
        return coordinates
    
    def t(self): # mutable
        delta = Lorenz(np.zeros(self.dimension))
        
        delta.state[0] = self.sigma * (self.state[1] - self.state[0])
        delta.state[1] = self.state[0] * (self.rho - self.state[2]) - self.state[1]
        delta.state[2] = self.state[0] * self.state[1] - self.beta * self.state[2]
        
        return delta
        
    def __add__(lhs, rhs):
        local_sum = Lorenz(np.zeros(lhs.dimension))
        
        local_sum.state = lhs.state + rhs.state
        local_sum.sigma = lhs.sigma + rhs.sigma
        local_sum.rho = lhs.rho + rhs.rho
        local_sum.beta = lhs.beta + rhs.beta
        
        return local_sum
    
    def __mul__(lhs, rhs):
        local_product = Lorenz(np.zeros(lhs.dimension))
        
        local_product.state = lhs.state * rhs
        local_product.sigma = lhs.sigma * rhs
        local_product.rho = lhs.rho * rhs
        local_product.beta = lhs.beta * rhs
        
        return local_product
    
    def integrate(self, dt):
        return integrand.integrate(self, dt)
        