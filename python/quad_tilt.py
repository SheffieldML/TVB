import numpy as np
import pylab as pb

from tilted import Tilted
from integrate import integrate

from IPython import parallel
from functools import partial

from likelihoods import student_t
import pylab

class quad_tilt(Tilted):
    def __init__(self, Y, in_parallel=False):
        """
        An illustration of quadtarture for use with var_EP.
        """
        Tilted.__init__(self,Y)
        self.Y = Y.flatten() # we're only doing 1D at the moment
        self.num_data = self.Y.size
        self.lik  = student_t() # hard coded right now. Incorporate into GPy when the code is ready.
        self._has_params = True
        self.num_params = 2

        self.parallel = in_parallel
        if self.parallel:
            self.client = parallel.Client()
            self.dv = self.client.direct_view()


    def _set_params(self, x):
        self.lik._set_params(x)

    def _get_params(self):
        return self.lik._get_params()

    def _get_param_names(self):
        return self.lik._get_param_names()

    def predictive_values(self, mu, var, percentiles):
        return self.lik.predictive_values(mu, var, percentiles)


    def set_cavity(self, mu, sigma2):
        """
        For a new series of cavity distributions,  compute the relevant moments and derivatives
        """
        Tilted.set_cavity(self, mu, sigma2)


        #quadrature!
        f = partial(integrate, lik=self.lik, derivs=self._has_params)
        if self.parallel:
            quads, numevals = zip(*self.dv.map(f, self.Y, self.mu, self.sigma))
        else:
            quads, numevals = zip(*map(f, self.Y, self.mu, self.sigma))
        quads = np.vstack(quads)

        self.Z = quads[:,0]
        self.mean = quads[:,1]/self.Z
        self.Ex2 = quads[:,2]/self.Z
        self.var = self.Ex2 - np.square(self.mean)
        self.Ex3 = quads[:,3]/self.Z
        self.Ex4 = quads[:,4]/self.Z
        self.dZ_dtheta = quads[:,5::3].T

        #derivatives of the mean, variance wrt theta
        self.dmean_dtheta = quads[:,6::3].T/self.Z - self.dZ_dtheta*self.mean/self.Z
        self.dEx2_dtheta = quads[:,7::3].T/self.Z  - self.dZ_dtheta*self.Ex2/self.Z
        self.dvar_dtheta = self.dEx2_dtheta - 2*self.mean*self.dmean_dtheta

        #derivatives of Z wrt cavity mean, var
        self.dZ_dmu = self.Z/self.sigma2*(self.mean - self.mu)
        self.dZ_dsigma2 = self.Z/self.sigma2**2/2*(self.Ex2 + self.mu**2 - 2*self.mu*self.mean) - 0.5*self.Z/self.sigma2

        #derivatives of the mean wrt cavity mean, var
        self.dmean_dmu = self.var/self.sigma2
        self.dmean_dsigma2 = -0.5*self.mean/self.sigma2 + 0.5*(self.Ex3 + self.mu**2*self.mean - 2*self.mu*self.Ex2)/self.sigma2**2 - self.mean*self.dZ_dsigma2/self.Z

        #derivatives of the variance wrt cavity mean, var
        self.dvar_dmu = (self.Ex3 - self.mean*self.Ex2)/self.sigma2 - 2.*self.mean*self.dmean_dmu
        self.dvar_dsigma2 = -0.5*self.Ex2/self.sigma2 + 0.5/self.sigma2**2*(self.Ex4 + self.Ex2*self.mu**2 - 2.*self.Ex3*self.mu) - self.Ex2*self.dZ_dsigma2/self.Z - 2*self.mean*self.dmean_dsigma2

    def pdf(self,X):
        lik = np.vstack([self.lik.pdf(x,y) for x, y in zip(X.T, self.Y)]).T
        cavity = np.exp(-0.5*np.log(2*np.pi) -0.5*np.log(self.sigma2) - np.square(X-self.mu)/self.sigma2)
        return lik*cavity

    def plot(self, index=0):
        pb.figure()
        xmin = self.mean[index] - 3*np.sqrt(self.var[index])
        xmax = self.mean[index] + 3*np.sqrt(self.var[index])
        xx = np.linspace(xmin, xmax, 100)
        pb.plot(xx, self.lik.pdf(xx, self.Y[index]),'b', label='likelihood')
        m = self.mu[index]
        s2 = self.sigma2[index]
        pb.plot(xx, np.exp(-0.5*np.square(xx-m)/s2)/np.sqrt(2*np.pi*s2),'g', label='prior')
        pb.plot(xx, self.lik.pdf(xx, self.Y[index])*np.exp(-0.5*np.square(xx-m)/s2)/np.sqrt(2*np.pi*s2)/self.Z[index],'r-', label='post')
        m = self.mean[index]
        s2 = self.var[index]
        pb.plot(xx, np.exp(-0.5*np.square(xx-m)/s2)/np.sqrt(2*np.pi*s2), 'r--', label='approx. posterior')

if __name__=='__main__':
    import GPy

    verbose = True

    # some data
    N = 800
    Y = np.random.randn(N)
    Y.sort()
    Y = np.sin(Y)
    
    # some cavity distributions:
    x = np.r_[-5:5:100j]
    mu = np.sin(x)
    var = np.exp(x * .3)#np.exp(np.random.randn(N))
    
    t = quad_tilt(Y,in_parallel=False)
    mu, q = t.predictive_values(mu, var, [20,80])

    pylab.figure()
    pylab.plot(x,mu,lw=2,color='g')
    pylab.fill_between(x, q[0], q[1],alpha=.3, color='k')
    pylab.fill_between(x, mu-1.281551565545*np.sqrt(var), mu+1.281551565545*np.sqrt(var),alpha=.3, color='r')


    import ipdb;ipdb.set_trace()

    #a base class for checking the gradient
    class cg(GPy.core.Model):
        def __init__(self,y, mu, var, in_parallel=True):
            self.tilted = quad_tilt(y,in_parallel=in_parallel)
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

    print 'grads of mean'
    c = cg_m(Y, mu, var)
    c.checkgrad(verbose=verbose)
    #[c.tilted.plot(i) for i in range(N)]
    c.parmap = False
    print c.checkgrad(verbose=verbose)

    print 'grads of var'
    c = cg_v(Y, mu, var)
    c.checkgrad(verbose=verbose)

    print 'grads of Z'
    c = cg_Z(Y, mu, var)
    c.checkgrad(verbose=verbose)

    #a different class for checking the erivatives of the parameters
    class cg(GPy.core.Model):
        def __init__(self,y, mu, var):
            self.mu, self.var = mu, var
            self.tilted = quad_tilt(y, in_parallel=True)
            self.tilted.set_cavity(mu, var)
            GPy.core.Model.__init__(self)
        def _set_params(self,x):
            self.tilted._set_params(x)
            self.tilted.set_cavity(self.mu, self.var)
        def _get_params(self):
            return self.tilted._get_params()
        def _get_param_names(self):
            return self.tilted._get_param_names()

    class cg_Z(cg):
        def log_likelihood(self):
            return self.tilted.Z.sum()
        def _log_likelihood_gradients(self):
            return self.tilted.dZ_dtheta.sum(1)

    print 'grads of Z wrt theta'
    c = cg_Z(Y, mu, var)
    c.checkgrad(verbose=verbose)

    class cg_m(cg):
        def log_likelihood(self):
            return self.tilted.mean.sum()
        def _log_likelihood_gradients(self):
            return self.tilted.dmean_dtheta.sum(1)

    print 'grads of mean wrt theta'
    c = cg_m(Y, mu, var)
    c.checkgrad(verbose=verbose)

    class cg_var(cg):
        def log_likelihood(self):
            return self.tilted.var.sum()
        def _log_likelihood_gradients(self):
            return self.tilted.dvar_dtheta.sum(1)

    print 'grads of var wrt theta'
    c = cg_var(Y, mu, var)
    c.checkgrad(verbose=verbose)

    class cg_logZ(cg):
        def log_likelihood(self):
            return np.sum(np.log(self.tilted.Z))
        def _log_likelihood_gradients(self):
            return np.sum(self.tilted.dZ_dtheta/self.tilted.Z,1)

    print 'grads of logZ wrt theta'
    c = cg_logZ(Y, mu, var)
    c.checkgrad(verbose=verbose)





