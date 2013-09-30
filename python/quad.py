import numpy as np
import pylab as pb
from scipy.integrate import quad

from tilted import tilted


class quad_tilt(Tilted):
    def __init__(self, Y):
        Tilted.__init__(self,Y)
        self.Y = Y



    def set_cavity(self, mu, sigma2):
        Tilted.set_cavity(self, mu, sigma2)


