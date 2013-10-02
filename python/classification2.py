import numpy as np
import pylab as pb
import GPy
from truncnorm import truncnorm
from GPy.models.gradient_checker import GradientChecker
from scipy.special import erf
import tilted

class classification(GPy.core.Model):
    def __init__(self, X, Y, kern=None):
        self.X = X
        self.Y = Y
        if kern is None:
            kern = GPy.kern.rbf(X.shape[1]) + GPy.kern.white(X.shape[1])
        self.kern = kern
        self.Y_sign = np.where(Y>0,1,-1)
        self.num_data, self.input_dim = self.X.shape
        self.no_K_grads_please = False
        GPy.core.Model.__init__(self)

        self.Ytilde = np.zeros(self.num_data)
        self.beta = np.zeros(self.num_data) + 0.1

        self.tilted = tilted.Heaviside(self.Y)

        self.ensure_default_constraints()
        self.constrain_positive('beta')

    def _set_params(self,x):
        self.Ytilde = x[:self.num_data]
        self.beta = x[self.num_data:2*self.num_data]
        self.kern._set_params_transformed(x[2*self.num_data:])

        #compute approximate posterior mean and variance - this is q(f) in RassWill notation,
        # and p(f | \tilde y) in ours
        self.K = self.kern.K(self.X)
        self.Ki, self.L, _,self.K_logdet = GPy.util.linalg.pdinv(self.K)
        self.Sigma_inv = self.Ki + np.diag(self.beta)
        self.Sigma,_,_,_ = GPy.util.linalg.pdinv(self.Sigma_inv)
        self.diag_Sigma = np.diag(self.Sigma)

        #TODO: use woodbury for inverse? We don't get Ki though :(
        #tmp = self.K + np.diag(1./self.beta)
        #L = GPy.util.linalg.jitchol(tmp)
        #LiK,_ = GPy.util.linalg.dtrtrs(L,self.K, lower=1)
        #self.Sigma_ = self.K - np.dot(LiK.T, LiK)
        #LiLiK,_ = GPy.util.linalg.dtrtrs(L, LiK, lower=1, trans=1)
        #self.SigmaKi_ = np.eye(self.num_data) - LiLiK.T

        self.mu = np.dot(self.Sigma, self.beta*self.Ytilde )

        #compute cavity means, vars (all at once!)
        self.cavity_vars = 1./(1./self.diag_Sigma - self.beta)
        self.cavity_means = self.cavity_vars * (self.mu/self.diag_Sigma - self.Ytilde*self.beta)

        #compute tilted distributions...
        self.tilted.set_cavity(self.cavity_means, self.cavity_vars)

    def _get_params(self):
        return np.hstack((self.Ytilde, self.beta, self.kern._get_params_transformed()))

    def _get_param_names(self):
        return ['Ytilde%i'%i for i in range(self.num_data)] +\
               ['beta%i'%i for i in range(self.num_data)] +\
               self.kern._get_param_names_transformed()


    def alternative_log_likelihood(self):
        """the log likelihood from classification1.py"""
        #expectation of log pseudo-likelihood times prior under q
        A = -self.num_data*np.log(2*np.pi) + 0.5*np.log(self.beta).sum() - 0.5*self.K_logdet
        A += -0.5*np.sum(self.beta*(np.square(self.Ytilde) + np.square(self.tilted.mean)  + self.tilted.var - 2.*self.tilted.mean*self.Ytilde))
        tmp, _ = GPy.util.linalg.dtrtrs(self.L,self.tilted.mean, lower=1)
        A += -0.5*np.sum(np.square(tmp)) - 0.5*np.sum(np.diag(self.Ki)*self.tilted.var)

        #entropy
        B = self.tilted.H.sum()

        #relative likelihood/ pseudo-likelihood normalisers
        C = np.log(self.tilted.Z).sum()
        D = (.5 * self.num_data * np.log(2 * np.pi)
              + np.sum(.5 * np.log(1. / self.beta + self.cavity_vars)
                       + .5 * (self.Ytilde - self.cavity_means) ** 2 / (1. / self.beta + self.cavity_vars)))
        return A + B + C + D

    def log_likelihood(self):
        #ignore log 2 pi terms, they cancel.

        #expectation of log prior under q
        tmp, _ = GPy.util.linalg.dtrtrs(self.L,self.tilted.mean, lower=1)
        A = -0.5*self.K_logdet -0.5*np.sum(np.square(tmp)) - 0.5*np.sum(np.diag(self.Ki)*self.tilted.var)

        #expectation of the (negative) log cavity
        B = 0.5*np.sum(np.log(self.cavity_vars)) + 0.5*np.sum(np.square(self.cavity_means - self.tilted.mean)/self.cavity_vars) + 0.5*np.sum(self.tilted.var/self.cavity_vars)

        #Z
        C = np.log(self.tilted.Z).sum()
        return A + B + C

    def _log_likelihood_gradients(self):
        """first compute gradients wrt cavity means/vars, then chain"""

        # partial derivatives: watch the broadcast!
        dcav_vars_dbeta = -(self.Sigma**2 / self.diag_Sigma**2 - np.eye(self.num_data) )*self.cavity_vars**2 # correct!
        dcav_means_dYtilde = (self.Sigma * self.beta[:, None] / self.diag_Sigma - np.diag(self.beta)) * self.cavity_vars # correct!

        dcav_means_dbeta = dcav_vars_dbeta * (self.mu / self.diag_Sigma - self.Ytilde * self.beta)
        tmp = self.Sigma / self.diag_Sigma
        dcav_means_dbeta += (tmp*(self.Ytilde[:,None] - self.mu[:,None]) + tmp**2*self.mu - np.diag(self.Ytilde))*self.cavity_vars

        #first compute gradietn wrt cavity parameters, then chain
        #A
        dA_dq_means = -np.dot(self.Ki, self.tilted.mean)
        dA_dq_vars = -0.5*np.diag(self.Ki)
        dA_dcav_means = dA_dq_vars*self.tilted.dvar_dmu + dA_dq_means*self.tilted.dmean_dmu
        dA_dcav_vars = dA_dq_vars*self.tilted.dvar_dsigma2
        dA_dcav_vars += dA_dq_means*self.tilted.dmean_dsigma2

        #B
        dB_dq_means = (self.tilted.mean - self.cavity_means)/self.cavity_vars
        dB_dq_vars = 0.5/self.cavity_vars
        dB_dcav_vars = 0.5/self.cavity_vars - 0.5*(np.square(self.cavity_means - self.tilted.mean) + self.tilted.var)/np.square(self.cavity_vars)
        dB_dcav_vars += dB_dq_means*self.tilted.dmean_dsigma2
        dB_dcav_vars += dB_dq_vars*self.tilted.dvar_dsigma2
        dB_dcav_means = (self.cavity_means - self.tilted.mean)/self.cavity_vars
        dB_dcav_means += dB_dq_vars*self.tilted.dvar_dmu + dB_dq_means*self.tilted.dmean_dmu

        #C
        dC_dcav_means = self.tilted.dZ_dmu/self.tilted.Z
        dC_dcav_vars = self.tilted.dZ_dsigma2/self.tilted.Z


        #sum gradients from all the different parts
        dL_dcav_vars = dA_dcav_vars + dB_dcav_vars + dC_dcav_vars
        dL_dcav_means = dA_dcav_means + dB_dcav_means + dC_dcav_means

        dL_dbeta = np.dot(dcav_means_dbeta, dL_dcav_means) + np.dot(dcav_vars_dbeta, dL_dcav_vars)
        dL_dYtilde = np.dot(dcav_means_dYtilde, dL_dcav_means)

        #ok, now gradient for K
        if self.no_K_grads_please:
            dL_dtheta = np.zeros(self.kern.num_params_transformed())
        else:
            #the symmetric parts
            tmp = dL_dcav_vars*np.square(self.cavity_vars/self.diag_Sigma)
            tmp += dL_dcav_means*(self.cavity_means - self.mu)*self.cavity_vars/np.square(self.diag_Sigma)
            KiSigma = np.dot(self.Ki, self.Sigma)

            tmp = KiSigma*tmp
            dL_dK = np.dot(tmp, KiSigma.T)

            #the non-symmetric parts
            dL_dK += (np.dot(self.Ki, self.mu)[:,None] * (dL_dcav_means*self.cavity_vars/self.diag_Sigma)[None,:]).dot(KiSigma.T)

            #the 'direct' part
            dL_dK -= .5 * self.Ki # for the log det.
            Kim = np.dot(self.Ki, self.tilted.mean)
            dL_dK += 0.5*Kim[:,None]*Kim[None,:]
            dL_dK += 0.5*np.dot(self.Ki*self.tilted.var, self.Ki)#the diag part

            dL_dtheta = self.kern.dK_dtheta(dL_dK, self.X)

        return np.hstack((dL_dYtilde, dL_dbeta, dL_dtheta))

    def _predict_raw(self, Xnew):
        """Predict the underlying GP function"""
        Kx = self.kern.K(Xnew, self.X)
        Kxx = self.kern.Kdiag(Xnew)
        L = GPy.util.linalg.jitchol(self.K + np.diag(1./self.beta))
        tmp, _ = GPy.util.linalg.dpotrs(L, self.Ytilde, lower=1)
        mu = np.dot(Kx, tmp)
        mu_ = np.dot(Kx, self.Ki).dot(self.mu)
        tmp, _ = GPy.util.linalg.dtrtrs(L, Kx.T, lower=1)
        var = Kxx - np.sum(np.square(tmp), 0)
        return mu, var

    def predict(self, Xnew):
        mu, var = self._predict_raw(Xnew)
        return 0.5*(1+erf(mu/np.sqrt(2.*var)))


    def plot_f(self):
        if self.X.shape[1]==1:
            pb.figure()
            pb.errorbar(self.X[:,0],self.Ytilde,yerr=2*np.sqrt(1./self.beta), fmt=None, label='approx. likelihood', ecolor='r')
            Xtest, xmin, xmax = GPy.util.plot.x_frame1D(self.X)
            mu, var = self._predict_raw(Xtest)
            GPy.util.plot.gpplot(Xtest, mu, mu - 2*np.sqrt(var), mu + 2*np.sqrt(var))
        elif self.X.shape[1]==2:
            pb.figure()
            Xtest,xx,yy, xymin, xymax = GPy.util.plot.x_frame2D(self.X)
            mu, var = self._predict_raw(Xtest)
            pb.contour(xx,yy,mu.reshape(*xx.shape))

    def plot(self):
        if self.X.shape[1]==1:
            pb.figure()
            Xtest, xmin, xmax = GPy.util.plot.x_frame1D(self.X)
            mu, var = self._predict_raw(Xtest)

            #GPy.util.plot.gpplot(Xtest, mu, mu - 2*np.sqrt(var), mu + 2*np.sqrt(var))
            pb.plot(self.X, self.Y, 'kx', mew=1)
            pb.plot(Xtest, 0.5*(1+erf(mu/np.sqrt(2.*var))), linewidth=2)
            pb.ylim(-.1, 1.1)
        elif self.X.shape[1]==2:
            pb.figure()
            Xtest,xx,yy, xymin, xymax = GPy.util.plot.x_frame2D(self.X)
            p = self.predict(Xtest)
            c = pb.contour(xx,yy,p.reshape(*xx.shape), [0.1, 0.25, 0.5, 0.75, 0.9], colors='k')
            pb.clabel(c)
            i1 = self.Y==1
            pb.plot(self.X[:,0][i1], self.X[:,1][i1], 'rx', mew=2, ms=8)
            i2 = self.Y==0
            pb.plot(self.X[:,0][i2], self.X[:,1][i2], 'wo', mew=2, mec='b')


    def natgrad(self):
        grads = self._log_likelihood_gradients()
        dL_dYtilde = grads[:self.num_data]
        dL_dbeta = grads[self.num_data:2*self.num_data]

        ll_old = self.log_likelihood()
        beta_old = self.beta.copy()
        Ytilde_old = self.Ytilde.copy()

        steplength = 1e-2
        for i in range(100):

            #which!?
            beta_new = self.beta + steplength*2.*np.diag(np.dot(self.Sigma_inv*dL_dbeta, self.Sigma_inv))
            beta_new = np.clip(beta_new, 1e-3, 1e3)


            By_new = self.beta*self.Ytilde + steplength*np.dot(self.Sigma_inv/self.beta.reshape(-1,1),dL_dYtilde)
            y_new = By_new/beta_new

            self.Ytilde = y_new
            self.beta = beta_new
            self._set_params(self._get_params())

            ll_new = self.log_likelihood()
            if (ll_new<ll_old) or np.isnan(ll_new):
                #step failed: reduce steplength and try again
                self.beta = beta_old
                self.Ytilde = Ytilde_old
                steplength /= 2.

                print i, ll_new, '(failed, reducing step length)'
            else:
                #sucess!
                print i, ll_new
                if (ll_new - ll_old) < 1e-6:
                    break # terminate
                ll_old = self.log_likelihood()
                beta_old = self.beta.copy()
                Ytilde_old = self.Ytilde.copy()

                steplength *= 1.1




if __name__=='__main__':
    pb.close('all')
    N = 20
    X = np.random.rand(N)[:,None]
    X = np.sort(X,0)
    Y = np.zeros(N)
    Y[X[:, 0] < 3. / 4] = 1.
    Y[X[:, 0] < 1. / 4] = 0.
    #Y = np.random.permutation(Y)
    #pb.plot(X[:, 0], Y, 'kx')
    from classification1 import classification as cls1
    m_ = cls1(X,Y)
    m_.optimize('bfgs')
    m = classification(X, Y, m_.kern.copy())
    m.constrain_positive('beta')
    #m.randomize();     m.checkgrad(verbose=True)
    m.optimize('bfgs')
    m.plot()
