import numpy as np
import pylab as pb
from scipy import stats

class qquad(object):
    def __init__(self,f,N=100):
        """
        a class for performing quadrature in 1D to compute difficult integrals of a distribution

        q(x) = N(x|mu, sigma2) f(x) /Z

        where Z is a normalising constant. We'll be making heavy use of Gauss-hermite quadrature!

        f should accept a numpy array and return one of the same shape (e.g. a numpy.ufunc)
        """
        self.f = f
        self.gh_x, self.gh_w = np.polynomial.hermite.hermgauss(N)
        self.gh_w /= self.gh_w.sum()
        self.N_gh_x = stats.norm.pdf(self.gh_x)

    def Z(self, mu, sigma2):
        sigma = np.sqrt(sigma2)
        #pb.plot(self.gh_x, self.gh_w, self.gh_x, self.f(self.gh_x*sigma + mu))
        return np.sum(self.f(self.gh_x*sigma + mu)*self.gh_w)

    def H(self, mu, sigma2):
        sigma = np.sqrt(sigma2)
        ff = self.f(self.gh_x*sigma + mu)
        Z = np.sum(ff*self.gh_w)*sigma
        H = np.sum(ff*np.log(ff*self.N_gh_x)*self.gh_w)*sigma
        return H

    def dH_dmu(self, mu, sigma2):
        pass #TODO

if __name__=='__main__':
    f = lambda x : np.where(x>0,1,0)
    q = qquad(f,200)
    from truncnorm import truncnorm
    mus = np.linspace(-2,2,100)
    tns = [truncnorm(mu, 6) for mu in mus]
    pb.plot(mus, [t.Z for t in tns],'bo')
    pb.plot(mus, [q.Z(mu, 6) for mu in mus],'ro')







