import numpy as np
import pylab as pb

class tilted:
    def __init__(self, Y):
        self.Y = Y
    def set_cavity(self, mu, sigma2):
        self.mu, self.sigma2 = mu, sigma2
        self.sigma = np.sqrt(self.sigma2)

class heaviside(tilted):
    def __init__(self, Y):
        tilted.__init__(Y)
        self.Ysign = np.where(self.Y==1,1,-1)

    def set_cavity(self, mu, sigma2):
        tilted.set_cavity(mu, sigma)
        self.a = self.Ysign*self.mu
        self.Z = norm_cdf(self.a)
        self.N = norm_pdf(self.a)
        self.N_Z = self.N/self.Z
        self.N_Z2 = np.square(self.N_Z)

        #compute moments
        self.mean = self.mu + self.Ysign*self.N_Z
        self.var = (1. - self.a * self.N_Z - self.N_Z2)

        #derivatives of moments
        self.dmean_dmu = 1. - (self.N_Z2 + self.a * self.N_Z)
        self.dmean_dsigma2 = self.Ysign * self.N_Z * 0.5/self.sigma * (1 + self.a * (self.a + self.N_Z))
        self.dvar_dmu = 
        self.dvar_dsigma2 = 

        #compute entropy
        self.H = 0.5*np.log(2*np.pi*self.sigma2) + np.log(self.Z) + 0.5*(self.mu**2 + self.var + self.mean**2 - 2*self.mean*self.mu)/self.sigma2

        #entropy derivatives
        self.dH_dmu = 
        self.dH_dsigma2



    


