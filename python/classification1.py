import numpy as np
import pylab as pb
import GPy
from truncnorm import truncnorm
from GPy.models.gradient_checker import GradientChecker

class classification(GPy.core.Model):
    def __init__(self, X, Y, kern):
        self.X = X
        self.Y = Y
        self.kern = kern
        self.Y_sign = np.where(Y>0,1,-1)
        self.num_data, self.input_dim = self.X.shape
        GPy.core.Model.__init__(self)

        self.Ytilde = np.zeros(self.num_data)
        self.beta = np.zeros(self.num_data) + 1

        self.ensure_default_constraints()

    def _set_params(self,x):
        self.Ytilde = x[:self.num_data]
        self.beta = x[self.num_data:2*self.num_data]
        self.kern._set_params_transformed(x[2*self.num_data:])

        #compute approximate posterior mean and variance - this is q(f) in RassWill notation,
        # and p(f | \tilde y) in ours
        self.K = self.kern.K(self.X)
        self.Ki, self.L, _,self.K_logdet = GPy.util.linalg.pdinv(self.K)
        self.Sigma,_,_,_ = GPy.util.linalg.pdinv(self.Ki + np.diag(self.beta))
        self.diag_Sigma = np.diag(self.Sigma)
        self.mu = np.dot(self.Sigma, self.beta*self.Ytilde )

        #compute cavity means, vars (all at once!)
        self.cavity_vars = 1./(1./self.diag_Sigma - self.beta)
        self.cavity_means = self.cavity_vars * (self.mu/self.diag_Sigma - self.Ytilde*self.beta)

        #compute q-distributions...
        self.truncnorms = [truncnorm(mu, var, ('left' if y==1 else 'right')) for mu, var, y in zip(self.cavity_means, self.cavity_vars, self.Y.flatten())]
        self.q_means = np.array([q.mean() for q in self.truncnorms])
        self.q_vars = np.array([q.var() for q in self.truncnorms])

    def _get_params(self):
        return np.hstack((self.Ytilde, self.beta, self.kern._get_params_transformed()))

    def _get_param_names(self):
        return ['Ytilde%i'%i for i in range(self.num_data)] +\
               ['beta%i'%i for i in range(self.num_data)] +\
               self.kern._get_param_names_transformed()

    def log_likelihood(self):
        #expectation of log pseudo-likelihood times prior under q
        A = -self.num_data*np.log(2*np.pi) + 0.5*np.log(self.beta).sum() - 0.5*self.K_logdet
        A += -0.5*np.sum(self.beta*(np.square(self.Ytilde) + np.square(self.q_means)  + self.q_vars - 2.*self.q_means*self.Ytilde))
        tmp, _ = GPy.util.linalg.dtrtrs(self.L,self.q_means, lower=1)
        A += -0.5*np.sum(np.square(tmp)) - 0.5*np.sum(np.diag(self.Ki)*self.q_vars)

        #entropy
        B = np.sum([q.H() for q in self.truncnorms])

        #relative likelihood/ pseudo-likelihood normalisers
        C = np.sum(np.log([q.Z for q in self.truncnorms]))
        D = -(.5 * self.num_data * np.log(2 * np.pi)
              + np.sum(.5 * np.log(1. / self.beta + self.cavity_vars)
                       + .5 * (self.Ytilde - self.cavity_means) ** 2 / (1. / self.beta + self.cavity_vars)))
#         return self.cavity_means[9]
        return A + B + C + D

    def _log_likelihood_gradients(self):
        """first compute gradients wrt cavity means/vars, then chain"""

        # partial derivatives: watch the broadcast!
        dcav_vars_dbeta = -(self.Sigma**2 / self.diag_Sigma**2 - np.eye(self.num_data) )*self.cavity_vars**2 # correct!
        #dcav_vars_dYtilde = 0
        dcav_means_dYtilde = (self.Sigma * self.beta[:, None] / self.diag_Sigma - np.diag(self.beta)) * self.cavity_vars # correct!

        dcav_means_dbeta = dcav_vars_dbeta * (self.mu / self.diag_Sigma - self.Ytilde * self.beta)
        tmp = self.Sigma / self.diag_Sigma
        dcav_means_dbeta += (tmp*(self.Ytilde[:,None] - self.mu[:,None]) + tmp**2*self.mu - np.diag(self.Ytilde))*self.cavity_vars

        #A
        dA_dYtilde =  self.beta * (self.q_means - self.Ytilde)
        dA_dbeta = 0.5/self.beta - 0.5*(np.square(self.Ytilde) + np.square(self.q_means) + self.q_vars -2.*self.q_means*self.Ytilde)
        dA_dq_means = self.beta*(self.Ytilde - self.q_means) - np.dot(self.Ki, self.q_means)
        dA_dq_vars = -0.5*(self.beta + np.diag(self.Ki))
        dA_dcav_vars = dA_dq_vars*np.array([q.dvar_dvar() for q in self.truncnorms])
        dA_dcav_vars += dA_dq_means*np.array([q.dmean_dvar() for q in self.truncnorms])
        dA_dcav_means = dA_dq_vars*np.array([q.dvar_dmu() for q in self.truncnorms])
        dA_dcav_means += dA_dq_means*np.array([q.dmean_dmu() for q in self.truncnorms])
        dA_dbeta += np.dot(dcav_means_dbeta, dA_dcav_means) + np.dot(dcav_vars_dbeta, dA_dcav_vars)
        dA_dYtilde += np.dot(dcav_means_dYtilde, dA_dcav_means)

        #B
        dB_dcav_means = np.array([q.dH_dmu() for q in self.truncnorms])
        dB_dcav_vars = np.array([q.dH_dvar() for q in self.truncnorms])
        dB_dbeta = np.dot(dcav_means_dbeta, dB_dcav_means) + np.dot(dcav_vars_dbeta, dB_dcav_vars)
        dB_dYtilde = np.dot(dcav_means_dYtilde, dB_dcav_means)

        #C
        dC_dcav_means = np.array([q.dZ_dmu()/q.Z for q in self.truncnorms])
        dC_dcav_vars = np.array([q.dZ_dvar()/q.Z for q in self.truncnorms])
        dC_dbeta = np.dot(dcav_means_dbeta, dC_dcav_means) + np.dot(dcav_vars_dbeta, dC_dcav_vars)
        dC_dYtilde = np.dot(dcav_means_dYtilde, dC_dcav_means)

        # D
        delta = np.eye(self.num_data)
        bv = (1. / self.beta + self.cavity_vars)
        ym = (self.Ytilde - self.cavity_means)
        dD_dYtilde = -np.sum(ym * (delta - dcav_means_dYtilde) / bv, 1)
        dD_dcav_means = np.sum(ym * delta / bv, 1)
        dD_dbeta = (-.5 * np.sum((dcav_vars_dbeta - delta / self.beta ** 2) / bv, 1)
                    + np.sum(.5 * ym ** 2 * ((dcav_vars_dbeta - (delta / self.beta ** 2)) / bv ** 2)
                             + ym * dcav_means_dbeta / (1. / self.beta + self.cavity_vars), 1))
        dD_dcav_vars = -.5 * np.sum((delta / bv) * (1. - (ym ** 2 / bv)), 1)

        #sum gradients from all the different parts
        dL_dbeta = dA_dbeta + dB_dbeta + dC_dbeta + dD_dbeta
        dL_dYtilde = dA_dYtilde + dB_dYtilde + dC_dYtilde + dD_dYtilde

        SigmaKi = self.Sigma.dot(self.Ki)
        # dcav_vars_dSigma = 1. / ((1. / self.diag_Sigma - self.beta) * self.diag_Sigma) ** 2
        vars_inv = 1. / (1 - self.diag_Sigma * self.beta)[:,None,None]
        dcav_vars_dK = ((vars_inv ** 2) * (SigmaKi[:, None, :] * SigmaKi[:, :, None]))

        dcav_means_dK = dcav_vars_dK * (self.mu / self.diag_Sigma - self.Ytilde * self.beta)[:, None, None]
        dmu_dK = (np.dot(self.Ki, self.mu)[None, None, :] * SigmaKi[:, :, None])  # correct!
        dSigma_inv_dK = (SigmaKi[:, None, :] * SigmaKi[:, :, None]) / self.diag_Sigma[:, None, None]  # correct!
        dcav_means_dK += vars_inv * (dmu_dK - self.mu[:, None, None] * dSigma_inv_dK)

        dL_dK = (((dA_dcav_vars + dB_dcav_vars + dC_dcav_vars + dD_dcav_vars)[:, None, None] * dcav_vars_dK).sum(0)
                 + ((dA_dcav_means + dB_dcav_means + dC_dcav_means + dD_dcav_means)[:, None, None] * dcav_means_dK).sum(0))
        dL_dK += -.5 * self.Ki.dot(np.eye(self.num_data) - (self.q_means[:, None].dot(self.q_means[None, :]) + np.eye(self.num_data) * self.q_vars).dot(self.Ki))

        return np.hstack((dL_dYtilde, dL_dbeta, self.kern.dK_dtheta(dL_dK, self.X)))

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


    def plot(self):
        pb.errorbar(self.X[:,0],self.Ytilde,yerr=2*np.sqrt(1./self.beta), fmt=None, label='approx. likelihood', ecolor='r')
        #pb.errorbar(self.X[:,0]+0.01,self.q_means,yerr=2*np.sqrt(self.q_vars), fmt=None, label='q(f) (non Gauss.)')
        pb.errorbar(self.X[:,0],self.mu,yerr=2*np.sqrt(np.diag(self.Sigma)), fmt=None, label='approx. posterior', ecolor='b')
        #pb.legend()
        Xtest, xmin, xmax = GPy.util.plot.x_frame1D(self.X)
        mu, var = self._predict_raw(Xtest)
        GPy.util.plot.gpplot(Xtest, mu, mu - 2*np.sqrt(var), mu + 2*np.sqrt(var))


if __name__=='__main__':
#     pb.close('all')
#     np.random.seed(1)
    N = 50
    X = np.random.rand(N)[:,None]
    X = np.sort(X,0)
    Y = np.where(X>0.5,1,0).flatten()
    #Y = np.random.permutation(Y)
    k = GPy.kern.rbf(1, .6, 0.2) + GPy.kern.white(1, 1e-1)
    m = classification(X, Y, k)
    m.constrain_positive('beta')
    # m.constrain_fixed('rbf')
    # m.constrain_fixed('white')
#     m['rbf_var'] = .5
#     m['rbf_len'] = .1
#     m['white'] = .1
#     m['beta'] = 1
#     m['Ytilde'] = 1

    m.checkgrad('.*|rbf|whit', verbose=True)
    m.optimize('bfgs', messages=1, max_iters=20)
#     m.plot()

#     mm = GPy.models.GPClassification(X,Y[:,None],kernel=k)
#     mm.constrain_fixed('')
#     mm.update_likelihood_approximation()
#     mm.plot_f()
#     pb.errorbar(mm.X[:,0],mm.likelihood.Y[:,0],yerr=2*np.sqrt(1./mm.likelihood.precision[:,0]), fmt=None, color='r')

    #     m.optimize('scg', messages=1, max_iters=200)
#     pb.figure(1); pb.clf()
#     m.plot()
#     pb.hlines(0, pb.xlim()[0], pb.xlim()[1], 'k', '--')
#
#     mm = GPy.models.GPClassification(X, Y[:, None], kernel=k)
#     mm.constrain_fixed('')
#     mm.update_likelihood_approximation()
#     pb.figure(2); pb.clf()
#     mm.plot_f(fignum=2)
#     pb.errorbar(mm.X[:, 0], mm.likelihood.Y[:, 0], yerr=2 * np.sqrt(1. / mm.likelihood.precision[:, 0]), fmt=None, color='r')
#     pb.hlines(0, pb.xlim()[0], pb.xlim()[1], 'k', '--')
#
#     pb.figure(3); pb.clf()
#     pb.scatter(m.X[:, 0], m._predict_raw(m.X)[0] > 0, color="r", s=40, marker='o', label="varEP", edgecolor="")
#     pb.scatter(mm.X[:, 0], mm._raw_predict(mm.X)[0] > 0, color="b", s=40, marker='o', label="EP", facecolor="")
#     pb.vlines(m.X[:, 0], 0, 1, 'k', '--', alpha=.3)
#     pb.legend(loc='center right', scatterpoints=1)
