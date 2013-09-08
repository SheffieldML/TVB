import numpy as np
import pylab as pb
from scipy.stats import norm as scipynorm
from scipy import special
pb.ion()
# pb.close('all')

class truncnorm:

    def __init__(self, mu, sigma2, side='left'):
        self.mu, self.sigma2, self.side  =  mu, sigma2, side
        self.sigma = np.sqrt(self.sigma2)
        self.compute_Z()

    def compute_Z(self):
        if self.side == 'left':
            self.Z = scipynorm.cdf(self.mu / self.sigma)
        if self.side == 'right':
            self.Z = scipynorm.cdf(-self.mu / self.sigma)

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
        self.mu = mean
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
        MV = np.vectorize(dmean_dvar)
        for m in range(-10,10,2):
            self.mu = m
            self.compute_Z()
            pb.figure('mean:'+str(m))
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
        N = scipynorm.pdf(self.mu/self.sigma)
        return  1 - (N/self.Z)**2 - self.mu*N/self.Z/self.sigma# this is right!

    def dvar_dmu(self):
        N = scipynorm.pdf(self.mu/self.sigma)
        return  -self.sigma*N/self.Z + self.mu**2*N/self.Z/self.sigma + 3*self.mu*N**2/self.Z**2 + 2*self.sigma*(N/self.Z)**3

    def dH_dmu(self):
        N = scipynorm.pdf(self.mu/self.sigma)
        return (0.5/self.sigma2)*(N/self.Z*(self.sigma + self.mu**2/self.sigma) + self.mu*N**2/self.Z**2)

    def dmean_dvar(self):
        a = self.mu/self.sigma
        N = scipynorm.pdf(a)
        N_Z = N/self.Z
        if self.side == 'right':
            a = -a
        dmean_dvar = N_Z * 1./(2*self.sigma) * (1 + a * (a + N_Z))
        if self.side == 'right':
            dmean_dvar = -dmean_dvar
        return dmean_dvar #N_Z*(np.square(self.mu/self.sigma) + 1 + N_Z*self.mu/self.sigma)/2/self.sigma # Okay!

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

if __name__ == '__main__':
    t = truncnorm(0, 3, side='left')
    #t.plot()
    #t.plot_dH_ds()
