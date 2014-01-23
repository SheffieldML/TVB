# Copyright (c) 2014, James Hensman, Max Zwiessele
# Distributed under the terms of the GNU General public License, see LICENSE.txt

import numpy as np
import pylab as pb
from scipy.stats import norm as scipynorm
from scipy import special
import itertools
from GPy.core.model import Model
pb.ion()
# pb.close('all')

class truncnorm:
    """
    a class to explore the properties of the truncated normal distribution.

    This isn;t actually used in the TVB algorithm in the end
    """
    def __init__(self, mu, sigma2, side='left'):
        self.mu, self.sigma2, self.side  =  mu, sigma2, side
        self.sigma = np.sqrt(self.sigma2)
        self.compute_Z()

    def compute_Z(self):
        if self.side == 'left':
            self.Z = scipynorm.cdf(self.mu / self.sigma)
        if self.side == 'right':
            self.Z = scipynorm.cdf(-self.mu / self.sigma)

    def dZ_dmu(self):
        if self.side == 'left':
            return scipynorm.pdf(self.mu/self.sigma)/self.sigma
        if self.side == 'right':
            return -scipynorm.pdf(self.mu/self.sigma)/self.sigma

    def dZ_dvar(self):
        if self.side == 'left':
            return -0.5*scipynorm.pdf(self.mu/self.sigma)*self.mu/self.sigma/self.sigma2
        if self.side == 'right':
            return 0.5*scipynorm.pdf(self.mu/self.sigma)*self.mu/self.sigma/self.sigma2

    def pdf(self,x):
        p = scipynorm.pdf(x,loc=self.mu,scale=self.sigma)
        if self.side=='left':
            return np.where(x>0,p,0)/self.Z
        if self.side=='right':
            return np.where(x<0,p,0)/self.Z
    
    def samples(self,n):
        """inefficient"""
        rvs = np.random.randn(10*n)*self.sigma + self.mu
        if self.side=='left':
            rvs = rvs[rvs>0]
        elif self.side=='right':
            rvs = rvs[rvs<0]
        if rvs.size < n:
            rvs = np.hstack((rvs, self.samples(n-rvs.size)))
        return rvs[:n]

    def plot(self,N=100):
        s = self.samples(N)
        xmin, xmax = s.min() - 0.1*(s.max()-s.min()), s.max() + 0.1*(s.max()-s.min())
        xx = np.linspace(xmin, xmax, 1000)
        pb.hist(s,100, normed=True)
        pb.plot(xx, self.pdf(xx), 'r', linewidth=2)
        pb.title('H=%f'%self.H_sample())

    def plot_dH_ds(self, mean=0):
        # self.mu = mean
        x = np.linspace(1E-2,5,1000)
        def h(s):
            self.sigma = np.sqrt(s)
            self.sigma2 = s
            self.compute_Z()
            return self.H()
        def dh(s):
            self.sigma = np.sqrt(s)
            self.sigma2 = s
            self.compute_Z()
            return self.dH_dvar()
        def dmean_dvar(s):
            self.sigma = np.sqrt(s)
            self.sigma2 = s
            self.compute_Z()
            return self.dmean_dvar()
        H = np.vectorize(h)
        dH = np.vectorize(dh)
        # MV = np.vectorize(dmean_dvar)
        for m in range(1):
#             self.mu = m
            self.compute_Z()
            pb.figure("mu:{:.4f} var:{:.4f} side:{:s}".format(self.mu, self.sigma, self.side))
            pb.clf()
            pb.plot(x,dH(x), label='dH', lw=1.5)
            #pb.plot(x,MV(x), label='dMV', lw=1.5)
            pb.plot(x[:-1], np.diff(H(x))/np.diff(x), label='numdH', lw=1.5)
            pb.legend()
            pb.twinx()
            pb.plot(x,H(x), label='H')

    def mean(self):
        if self.side=='left':
            return self.mu + self.sigma*scipynorm.pdf(self.mu/self.sigma)/self.Z
        if self.side=='right':
            return self.mu - self.sigma*scipynorm.pdf(self.mu/self.sigma)/self.Z

    def var(self):
        if self.side=='left':
            a = -self.mu/self.sigma
            Na = scipynorm.pdf(a)
            A = (1. + a*Na/self.Z - np.square(Na/self.Z))
            return self.sigma2*A
        if self.side=='right':
            b = -self.mu/self.sigma
            Nb = scipynorm.pdf(b)
            A = (1. + -b*Nb/self.Z - np.square(Nb/self.Z))
            return self.sigma2*A

    def H_sample(self,n=10000):
        """entropy by sampling"""
        s = self.samples(n)
        return -np.log(self.pdf(s)).mean()

    def H(self):
        H = 0.5*np.log(2*np.pi*self.sigma2) + np.log(self.Z)
        Ex = self.mean()
        H += 0.5*(self.mu**2 + self.var() + Ex**2 - 2*Ex*self.mu)/self.sigma2
        return H

    def dmean_dmu(self):
        a = self.mu / self.sigma
        N = scipynorm.pdf(a)
        if self.side == 'right':
            a = -a
        dmean_dmu_partial = (N / self.Z) ** 2 + a * N / self.Z
        return 1 - dmean_dmu_partial  # this is right!

    def dvar_dmu(self):
        mu = self.mu
        N = scipynorm.pdf(self.mu / self.sigma)
        if self.side == 'right':
            mu = -mu
            return self.sigma * N / self.Z - mu ** 2 * N / self.Z / self.sigma - 3 * mu * N ** 2 / self.Z ** 2 - 2 * self.sigma * (N / self.Z) ** 3
        # left:
        return -self.sigma * N / self.Z + mu ** 2 * N / self.Z / self.sigma + 3 * mu * N ** 2 / self.Z ** 2 + 2 * self.sigma * (N / self.Z) ** 3

    def dH_dmu(self):
        mu = self.mu
        N = scipynorm.pdf(mu / self.sigma)
        if self.side == 'right':
            mu = -mu
            return -((0.5 / self.sigma2) * (N / self.Z * (self.sigma + mu ** 2 / self.sigma) + mu * N ** 2 / self.Z ** 2))
        # left:
        return (0.5 / self.sigma2) * (N / self.Z * (self.sigma + mu ** 2 / self.sigma) + mu * N ** 2 / self.Z ** 2)

    def dmean_dvar(self):
        a = self.mu/self.sigma
        N = scipynorm.pdf(a)
        N_Z = N/self.Z
        if self.side == 'right':
            a = -a
        half_sigma_inv = 1./(2.*self.sigma)
        dmean_dvar = N_Z * half_sigma_inv * (1 + a * (a + N_Z))
        if self.side == 'right':
            dmean_dvar = -dmean_dvar
        return dmean_dvar  # N_Z * (np.square(self.mu / self.sigma) + 1 + N_Z * self.mu / self.sigma) / 2 / self.sigma  # Okay!

    def dvar_dvar(self):
        """ The derivative of the truncated variance wrt the Gaussian variance..."""
        if self.side == 'left':
            a = self.mu / self.sigma; a2 = a**2
        elif self.side == 'right':
            a = -self.mu / self.sigma; a2 = a**2
        N = scipynorm.pdf(a)
        N_Z = N/self.Z
        return (1 - N_Z * (N_Z + a * (.5 + .5*a2 + N_Z * (1.5*a + N_Z))))

    def dH_dvar(self):
        mu, sigma = self.mu, self.sigma
        a = mu / sigma
        if self.side == 'right':
            a=-a
        Ex = self.mean()
        N_Z = scipynorm.pdf(a) / self.Z; #N_Z2 = np.square(N_Z)
        dA_dvar = 1./(2*sigma**2) - N_Z * (a/(2*sigma**2))
        dmean_dvar = self.dmean_dvar()
        dvar_dvar = self.dvar_dvar()
        return (
                dA_dvar
                + .5 * (1./(sigma**2)) * (dvar_dvar + (2*Ex - 2*mu) * dmean_dvar)
                - 0.5/(sigma**4) * (mu**2 + self.var() + Ex**2 - 2*Ex*mu)
                )

class TestTruncnorm(Model, truncnorm):
    def __init__(self, mu, sigma2, side):
        self.mu, self.sigma2, self.side = mu, sigma2, side
        self.sigma = np.sqrt(self.sigma2)
        self.compute_Z()
        super(TestTruncnorm, self).__init__()
    def _get_param_names(self):
        return ['mu', 'var']
    def _set_params(self, x):
        self.mu = float(x[0])
        self.sigma = float(x[1])
        self.sigma2 = np.square(self.sigma)
        # self.sigma = np.sqrt(self.sigma2)
        self.compute_Z()
    def _get_params(self):
        self.compute_Z()
        return np.array([self.mu, self.sigma])
    def log_likelihood(self):
        self.compute_Z()
        #return self.H()
    def _log_likelihood_gradients(self):
        self.compute_Z()
        # return np.array([self.dH_dmu(), self.dH_dvar()])
        return np.array([self.dH_dmu(), self.dH_dvar() * 2 * self.sigma])

class TestTruncnorm2(Model, truncnorm):
    def __init__(self, mu, sigma2, side):
        self.mu, self.sigma2, self.side = mu, sigma2, side
        self.sigma = np.sqrt(self.sigma2)
        self.compute_Z()
        super(TestTruncnorm2, self).__init__()
    def _get_param_names(self):
        return ['mu', 'var']
    def _set_params(self, x):
        self.mu = float(x[0])
        self.sigma2 = float(x[1])
        self.sigma = np.sqrt(self.sigma2)
        self.compute_Z()
    def _get_params(self):
        self.compute_Z()
        return np.array([self.mu, self.sigma2])
    def log_likelihood(self):
        self.compute_Z()
        return self.Z
    def _log_likelihood_gradients(self):
        self.compute_Z()
        return np.array([self.dZ_dmu(), self.dZ_dvar()])
if __name__ == '__main__':
    mu, sigma = np.random.randn(), np.random.rand()
    t_left = TestTruncnorm2(mu, sigma, 'left')
    t_right = TestTruncnorm2(mu, sigma, 'right')
