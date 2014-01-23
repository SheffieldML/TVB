# Copyright (c) 2014, James Hensman
# Distributed under the terms of the GNU General public License, see LICENSE.txt

import numpy as np
import pylab as pb
import GPy
from truncnorm import truncnorm
from scipy.special import erf
import tilted

class classification(GPy.core.Model):
    """

    A first stab at a *sparse* classification method.

    """
    def __init__(self, X, Y, Z=None, kern=None):
        self.X = X
        self.Y = Y
        self.Y_sign = np.where(Y>0,1,-1)
        if kern is None:
            kern = GPy.kern.rbf(X.shape[1]) + GPy.kern.white(X.shape[1])
        self.kern = kern

        if Z is None:
            i = np.random.permutation(X.shape[0])[:10]
            Z = X[i,:]
        self.Z = Z
        assert Z.shape[1] == X.shape[1], "bad inducing variable shape"
        self.num_data, self.input_dim = self.X.shape
        self.num_inducing = Z.shape[0]
        self.no_K_grads_please = False
        GPy.core.Model.__init__(self)

        self.Ytilde = np.zeros(self.num_inducing)
        self.beta = np.zeros(self.num_inducing) + 0.1

        self.tilted = tilted.Heaviside(self.Y)

        self.ensure_default_constraints()
        self.constrain_positive('beta')

    def _set_params(self,x):
        self.Ytilde = x[:self.num_inducing]
        self.beta = x[self.num_inducing:2*self.num_inducing]
        self.Z = x[2*self.num_inducing:2*self.num_inducing + self.num_inducing*self.input_dim].reshape(self.num_inducing, self.input_dim)
        self.kern._set_params_transformed(x[2*self.num_inducing + self.num_inducing*self.input_dim:])

        #compute the inverse in the prior...yuck we should remove this. 
        self.K = self.kern.K(self.X)
        self.Ki, self.L, _, self.K_logdet = GPy.util.linalg.pdinv(self.K)

        #compute approximate posterior mean and variance q(u) in RassWill notation,
        # and p(u | \tilde y) in ours
        self.Kmm = self.kern.K(self.Z)
        self.Kmn = self.kern.K(self.Z, self.X)

        #TODO: might need these for prediction!
        #self.Kmmi, self.Lm, _,self.Kmm_logdet = GPy.util.linalg.pdinv(self.Kmm)
        #self.Sigma_inv = self.Ki + np.diag(self.beta)
        #self.Sigma,_,_,_ = GPy.util.linalg.pdinv(self.Sigma_inv)
        #self.mu = np.dot(self.Sigma, self.beta*self.Ytilde )


        #compute cavity means, vars (all at once!)
        self.Ai, self.LA, _,self.A_logdet = GPy.util.linalg.pdinv(self.Kmm + np.diag(1./self.beta))
        # note that these aren't the 'cavity' so much as the 'marginals of the approximation'
        self.Aiy, _ = GPy.util.linalg.dpotrs(self.LA, self.Ytilde, lower=1)
        self.cavity_means = np.dot(self.Kmn.T, self.Aiy)
        tmp, _ = GPy.util.linalg.dtrtrs(self.LA, self.Kmn, lower=1)
        self.cavity_vars = self.kern.Kdiag(self.X) - np.sum(np.square(tmp),0)

        #compute tilted distributions...
        self.tilted.set_cavity(self.cavity_means, self.cavity_vars)

    def _get_params(self):
        return np.hstack((self.Ytilde, self.beta, self.Z.flatten(), self.kern._get_params_transformed()))

    def _get_param_names(self):
        return ['Ytilde%i'%i for i in range(self.num_inducing)] +\
               ['beta%i'%i for i in range(self.num_inducing)] +\
               sum([['iip_%i_%i' % (i, j) for j in range(self.Z.shape[1])] for i in range(self.Z.shape[0])], []) + \
               self.kern._get_param_names_transformed()


    def log_likelihood(self):
        #perhaps unsurprisingly, this is identical to classification2...
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
        AiKmn = np.dot(self.Ai, self.Kmn)
        dcav_means_dYtilde = AiKmn
        #dcav_var_dYtilde = 0.
        BiAiKmn = (1./self.beta)[:,None]*AiKmn
        BiAiy = self.Aiy/self.beta
        #dcav_means_dbeta = (BiAiKmn.T*BiAiy).T
        dcav_means_dbeta = BiAiy[:,None]*BiAiKmn
        dcav_vars_dbeta = -BiAiKmn**2


        #first compute gradietn wrt cavity parameters, then chain
        #A
        dA_dq_means = -np.dot(self.Ki, self.tilted.mean)
        dA_dq_vars = -0.5*np.diag(self.Ki)
        dA_dcav_means = dA_dq_vars*self.tilted.dvar_dmu + dA_dq_means*self.tilted.dmean_dmu
        dA_dcav_vars = dA_dq_vars*self.tilted.dvar_dsigma2 + dA_dq_means*self.tilted.dmean_dsigma2

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
        tmp = np.dot(self.Ki, self.tilted.mean[:,None])
        dL_dKnn = -0.5*self.Ki + 0.5*np.dot(tmp, tmp.T) + 0.5*np.dot(self.Ki*self.tilted.var, self.Ki)

        dL_dKmn = np.dot(self.Aiy[:,None], dL_dcav_means[None,:]) # chain through cav means
        dL_dKmn += -2.*AiKmn*dL_dcav_vars[None,:]

        dL_dKmm = np.dot(AiKmn*dL_dcav_vars, AiKmn.T)
        dL_dKmm -= np.dot(self.Aiy[:,None], np.sum(AiKmn*dL_dcav_means,1)[None,:])

        dL_dtheta = self.kern.dK_dtheta(dL_dKnn, self.X)
        dL_dtheta += self.kern.dK_dtheta(dL_dKmm, self.Z)
        dL_dtheta += self.kern.dK_dtheta(dL_dKmn, self.Z, self.X)
        dL_dtheta += self.kern.dKdiag_dtheta(dL_dcav_vars, self.X)

        dL_dZ = self.kern.dK_dX(dL_dKmn, self.Z, self.X).flatten()
        dL_dZ += self.kern.dK_dX(dL_dKmm, self.Z).flatten()

        return np.hstack((dL_dYtilde, dL_dbeta, dL_dZ, dL_dtheta))

    def _predict_raw(self, Xnew):
        """Predict the underlying GP function"""
        Kx = self.kern.K(Xnew, self.Z)
        Kxx = self.kern.Kdiag(Xnew)
        mu = np.dot(Kx, self.Aiy)
        tmp, _ = GPy.util.linalg.dtrtrs(self.LA, Kx.T, lower=1)
        var = Kxx - np.sum(np.square(tmp), 0)
        return mu, var

    def predict(self, Xnew):
        mu, var = self._predict_raw(Xnew)
        return 0.5*(1+erf(mu/np.sqrt(2.*var)))


    def plot_f(self):
        if self.X.shape[1]==1:
            pb.figure()
            pb.errorbar(self.Z[:,0],self.Ytilde,yerr=2*np.sqrt(1./self.beta), fmt=None, label='approx. likelihood', ecolor='r')
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
            pb.plot(self.Z.flatten(), np.zeros(self.num_inducing)-0.1, 'r|', ms=20, mew=2)
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
    N = 50
    M = 20
    X = np.random.rand(N)[:,None]
    Z = np.random.rand(M)[:,None]
    X = np.sort(X,0)
    Y = np.zeros(N)
    Y[X[:, 0] < 3. / 4] = 1.
    Y[X[:, 0] < 1. / 4] = 0.
    from classification2 import classification as cls2
    m_ = cls2(X,Y)
    #m_.optimize()
    m = classification(X, Y, Z, m_.kern.copy())
    m.checkgrad(verbose=1)
    m.constrain_positive('beta')
    m.constrain_fixed('(rbf|whi)')
    m.constrain_fixed('iip')
    #m.randomize();     m.checkgrad(verbose=True)
    m.optimize('scg', messages=1)
    m.plot()
    m.checkgrad(verbose=1)
