import numpy as np
import pylab as pb
pb.close('all')
from scipy.integrate import quad
from scipy.special import gamma, digamma

from tilted import Tilted
from quadvgk import inf_quadvgk

SQRT_2PI = np.sqrt(2.*np.pi)
LOG_SQRT_2PI = np.log(SQRT_2PI)

class student_t():
    def __init__(self):
        self._set_params(np.ones(2))
    def _set_params(self, p):
        self.nu, self.lamb = p
    def _get_params(self):
        return np.array([self.nu, self.lamb])
    def _get_param_names(self):
        return ['nu', 'lambda']
    def pdf(self, x, Y):
        x2 = np.square(x-Y)
        return gamma((self.nu + 1)/2.) / gamma(self.nu/2.) * np.sqrt(self.lamb/(self.nu*np.pi) ) * np.power(1 + self.lamb*x2/self.nu, -(self.nu + 1.)/2.)
    def dlnpdf_dtheta(self, x, Y):
        x2 = np.square(x-Y)
        dnu = 0.5*digamma((self.nu + 1.)/2.) - 0.5*digamma(self.nu/2.) - 0.5/self.nu - 0.5*np.log(1. + self.lamb*x2/self.nu) + 0.5*(self.nu + 1.)*(self.lamb*x2/self.nu**2)/(1. + self.lamb*x2/self.nu)
        dlamb =  0.5/self.lamb - 0.5*(self.nu + 1.)*(x2/self.nu/(1.+self.lamb*x2/self.nu))
        return np.vstack((dnu, dlamb))


class quad_tilt(Tilted):
    def __init__(self, Y):
        """
        An illustration of quadtarture for use with var_EP.
        """
        Tilted.__init__(self,Y)
        self.Y = Y.flatten() # we're only doing 1D at the moment
        self.lik  = student_t() # hard coded right now. Incorporate into GPy when the code is ready.
        self._has_params = True
        self.num_params = 2

    def _set_params(self, x):
        self.lik._set_params(x)
    def _get_params(self):
        return self.lik._get_params()
    def _get_param_names(self):
        return self.lik._get_param_names()


    def integrands(self, lik, Y, m, s, derivs=True):
        """
        compute the multiple-function of the integrands we want

        This function returns a function!

        f accepts a vector of inputs (1D numpy array), representing points of
        the variable of integration.  f returns a matrix representing several
        functions evaluated at those points. Let c(x) be a the 'cavity'
        distribution, c(x) = N(x|m, s**2), p(y|x) is a likelihood, then f
        computes the following functions on x
        p(y|x) * c(x)
        p(y|x) * c(x) * x
        p(y|x) * c(x) * x**2
        p(y|x) * c(x) * x**3
        p(y|x) * c(x) * x**4

        if derivs is true, we also stack in:

        p(y|x) * c(x) * d(ln p(y|x) / d theta)
        p(y|x) * c(x) * d(ln p(y|x) / d theta) + x
        p(y|x) * c(x) * d(ln p(y|x) / d theta) + x**2

        where theta is some parameter of the likelihood (e.g. the std of the noise, of the degree of freedom)

        If there are several parameters which require derivatives, then we have
        multiple lines for each.

        """
        assert np.array(Y).size==1, "we're only doing 1 data point at a time"
        if derivs:
            def f(x):
                a = lik.pdf(x, Y) * np.exp(-0.5*np.square((x-m)/s))/SQRT_2PI/s
                p = np.power(x, np.arange(5)[:,None])
                pp = np.tile(p[:3], [self.num_params, 1])
                derivs = lik.dlnpdf_dtheta(x, Y).repeat(3,0)
                return a * np.vstack((p, pp*derivs))[:,None]
        else:
            def f(x):
                return lik(x) * np.exp(-0.5*np.square((x-m)/s))/SQRT_2PI/s * np.power(x, np.arange(5))[:,None]
        return f

    def set_cavity(self, mu, sigma2):
        """
        For a new series of cavity distributions,  compute the relevant moments and derivatives
        """
        Tilted.set_cavity(self, mu, sigma2)

        #quadrature!
        #TODO: parallelise this loop (optionally?)
        quads = np.vstack([inf_quadvgk(self.integrands(self.lik, y_i, m, s, self._has_params))[0] for y_i, m, s in zip(self.Y, self.mu, self.sigma)])
        self.quads = quads
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

    # some data
    N = 5
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

    print 'grads of mean'
    c = cg_m(Y, mu, var)
    c.checkgrad(verbose=1)
    [c.tilted.plot(i) for i in range(N)]

    print 'grads of var'
    c = cg_v(Y, mu, var)
    c.checkgrad(verbose=1)

    print 'grads of Z'
    c = cg_Z(Y, mu, var)
    c.checkgrad(verbose=1)

    #a different class for checking the erivatives of the parameters
    class cg(GPy.core.Model):
        def __init__(self,y, mu, var):
            self.mu, self.var = mu, var
            self.tilted = quad_tilt(y)
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
    c.checkgrad(verbose=1)

    class cg_m(cg):
        def log_likelihood(self):
            return self.tilted.mean.sum()
        def _log_likelihood_gradients(self):
            return self.tilted.dmean_dtheta.sum(1)

    print 'grads of mean wrt theta'
    c = cg_m(Y, mu, var)
    c.checkgrad(verbose=1)

    class cg_var(cg):
        def log_likelihood(self):
            return self.tilted.var.sum()
        def _log_likelihood_gradients(self):
            return self.tilted.dvar_dtheta.sum(1)

    print 'grads of var wrt theta'
    c = cg_var(Y, mu, var)
    c.checkgrad(verbose=1)

    class cg_logZ(cg):
        def log_likelihood(self):
            return np.sum(np.log(self.tilted.Z))
        def _log_likelihood_gradients(self):
            return np.sum(self.tilted.dZ_dtheta/self.tilted.Z,1)

    print 'grads of logZ wrt theta'
    c = cg_logZ(Y, mu, var)
    c.checkgrad(verbose=1)





