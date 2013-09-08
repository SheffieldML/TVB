import numpy as np
import pylab as pb
import GPy

class grad1(GPy.core.Model):
    """
    A dumb class to make sure I got all my partial derivatives right...
    """
    def __init__(self):
        self.X = np.sort(np.random.randn(10,1))
        self.kern = GPy.kern.rbf(1) + GPy.kern.white(1,1e-3)
        self.K = self.kern.K(self.X)

        #initialise params
        self.beta = np.random.randn(10)

    def _get_params(self):
        return self.beta

    def _get_param_names(self):
        return 
        
