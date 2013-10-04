import numpy as np
import pylab as pb
from scipy.integrate import quad

from tilted import Tilted

SQRT_2PI = np.sqrt(2.*np.pi)
LOG_SQRT_2PI = np.log(SQRT_2PI)


class quad_tilt(Tilted):
    def __init__(self, Y):
        Tilted.__init__(self,Y)
        self.Y = Y.flatten() # we're only doing 1D at the moment

        #Cauchy hard-coded for now
        def cauchy(y):
            def f(x):
                return 1./np.pi/(1. + np.square(x-y))
            return f
        self.likelihoods = [cauchy(y) for y in self.Y]

    #a series of function generators. Lmabda don't work for this, sadly :(
    def lik_cavity(self, lik, m, s):
        def f(x):
            return lik(x) * np.exp(-0.5*np.square((x-m)/s))/SQRT_2PI/s
        return f

    def x_lik_cav(self, lik, m, s):
        def f(x):
            return x * lik(x) * np.exp(-0.5*np.square((x-m)/s))/SQRT_2PI/s
        return f

    def x2_lik_cav(self, lik, m, s):
        def f(x):
            return x**2 * lik(x) * np.exp(-0.5*np.square((x-m)/s))/SQRT_2PI/s
        return f

    def x3_lik_cav(self, lik, m, s):
        def f(x):
            return x**3 * lik(x) * np.exp(-0.5*np.square((x-m)/s))/SQRT_2PI/s
        return f

    def x4_lik_cav(self, lik, m, s):
        def f(x):
            return x**4 * lik(x) * np.exp(-0.5*np.square((x-m)/s))/SQRT_2PI/s
        return f

    def set_cavity(self, mu, sigma2):
        Tilted.set_cavity(self, mu, sigma2)

        #quadrature!
        self.Z = np.array([quad(self.lik_cavity(f, m, s) , -np.inf, np.inf)[0] for f, m, s in zip(self.likelihoods, self.mu, self.sigma)])
        self.mean_unnorm = np.array([quad(self.x_lik_cav(f, m, s), -np.inf, np.inf)[0] for f, m, s in zip(self.likelihoods, self.mu, self.sigma)])
        self.mean = self.mean_unnorm / self.Z
        self.Ex2 = np.array([quad(self.x2_lik_cav(f, m, s), -np.inf, np.inf)[0] for f, m, s in zip(self.likelihoods, self.mu, self.sigma)])
        self.Ex2 /= self.Z
        self.Ex3 = np.array([quad(self.x3_lik_cav(f, m, s), -np.inf, np.inf)[0] for f, m, s in zip(self.likelihoods, self.mu, self.sigma)])
        self.Ex3 /= self.Z
        self.Ex4 = np.array([quad(self.x4_lik_cav(f, m, s), -np.inf, np.inf)[0] for f, m, s in zip(self.likelihoods, self.mu, self.sigma)])
        self.Ex4 /= self.Z

        self.dZ_dmu = self.Z/self.sigma2*(self.mean - self.mu)
        self.dZ_dsigma2 = self.Z/self.sigma2**2/2*(self.Ex2 + self.mu**2 - 2*self.mu*self.mean) - 0.5*self.Z/self.sigma2

        #get the variance throught the square rule
        self.var = self.Ex2 - np.square(self.mean)

        #derivatives of the mean
        self.dmean_dmu = self.var/self.sigma2

        self.dmean_dsigma2 = -0.5*self.mean/self.sigma2 + 0.5*(self.Ex3 + self.mu**2*self.mean - 2*self.mu*self.Ex2)/self.sigma2**2 - self.mean*self.dZ_dsigma2/self.Z

        #derivatives of the variance
        self.dvar_dmu = (self.Ex3 - self.mean*self.Ex2)/self.sigma2 - 2.*self.mean*self.dmean_dmu

        self.dvar_dsigma2 = -0.5*self.Ex2/self.sigma2 + 0.5/self.sigma2**2*(self.Ex4 + self.Ex2*self.mu**2 - 2.*self.Ex3*self.mu) - self.Ex2*self.dZ_dsigma2/self.Z - 2*self.mean*self.dmean_dsigma2

        #entropy
        #tmp = np.array([quad(self.lik_cav_log_lik_cav(f, m, s), -np.inf, np.inf)[0] for f, m, s in zip(self.likelihoods, self.mu, self.sigma)])
        #self.entropy = (tmp - np.log(self.Z))/self.Z

        #self.dH_dmu = ??
        #self.dH_dsigma2 = ??



    def plot(self, index=0):
        pb.figure()
        xmin = self.mean[index] - 3*np.sqrt(self.var[index])
        xmax = self.mean[index] + 3*np.sqrt(self.var[index])
        xx = np.linspace(xmin, xmax, 100)
        pb.plot(xx, self.likelihoods[index](xx),'b', label='likelihood')
        m = self.mu[index]
        s2 = self.sigma2[index]
        pb.plot(xx, np.exp(-0.5*np.square(xx-m)/s2)/np.sqrt(2*np.pi*s2),'g', label='prior')
        pb.plot(xx, self.lik_cavity(self.likelihoods[index], self.mu[index], self.sigma[index])(xx)/self.Z[index], 'r-', label='posterior')

        m = self.mean[index]
        s2 = self.var[index]
        pb.plot(xx, np.exp(-0.5*np.square(xx-m)/s2)/np.sqrt(2*np.pi*s2), 'r--', label='approx. posterior')

if __name__=='__main__':
    import GPy

    # some data
    N = 10
    Y = np.random.randn(N)

    # some cavity distributions:
    mu = np.random.randn(N)*10
    var = np.exp(np.random.randn(N))

    #a base class for checking the gradient
    class cg(GPy.core.Model):
        def __init__(self,y, mu, var):
            self.tilted = quad_tilt(y)
            self.tilted.set_cavity(mu, var)
            self.N = y.size
            GPy.core.Model.__init__(self)
        def _set_params(self,x):
            self.tilted.set_cavity(x[:self.N], x[self.N:])
        def _get_params(self):
            return np.hstack((self.tilted.mu, self.tilted.sigma2))
        def _get_param_names(self):
            return ['mu_%i'%i for i in range(self.N)] +  ['sigma2_%i'%i for i in range(self.N)]

    class cg_Z(cg):
        def log_likelihood(self):
            return self.tilted.Z.sum()
        def _log_likelihood_gradients(self):
            return np.hstack((self.tilted.dZ_dmu, self.tilted.dZ_dsigma2))

    class cg_m(cg):
        def log_likelihood(self):
            return self.tilted.mean.sum()
        def _log_likelihood_gradients(self):
            return np.hstack((self.tilted.dmean_dmu, self.tilted.dmean_dsigma2))

    class cg_v(cg):
        def log_likelihood(self):
            return self.tilted.var.sum()
        def _log_likelihood_gradients(self):
            return np.hstack((self.tilted.dvar_dmu, self.tilted.dvar_dsigma2))

    class cg_H(cg):
        def log_likelihood(self):
            return self.tilted.H.sum()
        def _log_likelihood_gradients(self):
            return np.hstack((self.tilted.dH_dmu, self.tilted.dH_dsigma2))


    #print 'grads of mean'
    #c = cg_m(Y, mu, var)
    #c.checkgrad(verbose=1)

    #print 'grads of var'
    #c = cg_v(Y, mu, var)
    #c.checkgrad(verbose=1)

    #print 'grads of H'
    #c = cg_H(Y, mu, var)
    #c.checkgrad(verbose=1)

    print 'grads of Z'
    c = cg_Z(Y, mu, var)
    c.checkgrad(verbose=1)



            



