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
        # A += -0.5*np.sum(self.beta*(np.square(self.Ytilde - self.q_means) + self.q_vars))
        A += -0.5 * np.sum(self.beta * (np.square(self.Ytilde) + self.q_vars))
        tmp, _ = GPy.util.linalg.dtrtrs(self.L,self.q_means, lower=1)
        A += -0.5*np.sum(np.square(tmp)) - 0.5*np.sum(np.diag(self.Ki)*self.q_vars)

        #entropy
        B = np.sum([q.H() for q in self.truncnorms])

        #relative likelihood/ pseudo-likelihood normalisers
        C = np.sum(np.log([q.Z for q in self.truncnorms]))
        #C += - ??
        #return A + B + C
        return B

    def _log_likelihood_gradients(self):
        # first compute gradients wrt cavity means/vars, then chain
        dq_dcav_means = np.array([q.dH_dmu() for q in self.truncnorms])
        dq_dcav_vars = np.array([q.dH_dvar() for q in self.truncnorms])

        #A TODO
        dA_dYtilde = 0  # self.beta * (self.Ytilde - self.q_means)
        dA_dbeta = 0

        #B
        #watch the broadcast!
        dcav_vars_dbeta = -(self.Sigma**2 / self.diag_Sigma**2 - np.eye(self.num_data) )*self.cavity_vars**2
        dcav_means_dbeta_l = dcav_vars_dbeta * (self.mu / self.diag_Sigma - self.Ytilde * self.beta)
        dmu_dbeta = (self.Ytilde[:, None] - self.cavity_means) * self.Sigma  # np.eye(self.num_data)#TODO
        dcav_means_dbeta_r = self.cavity_vars * ((dmu_dbeta / self.diag_Sigma + self.mu * (self.Sigma / self.diag_Sigma) ** 2) - (self.Ytilde * np.eye(self.beta.shape[0])))
        dcav_means_dbeta = dcav_means_dbeta_l + dcav_means_dbeta_r

        dcav_means_dbeta = self.cavity_vars ** 2 * ((self.Sigma / self.diag_Sigma) ** 2 - np.eye(self.beta.shape[0])) * ((self.mu / self.diag_Sigma) - self.Ytilde * self.beta)
        dcav_means_dbeta += self.cavity_vars * self.Sigma / self.diag_Sigma * (self.Ytilde[:, None] - self.mu[:, None])
        dcav_means_dbeta += self.mu * self.cavity_vars * (self.Sigma / self.diag_Sigma) ** 2
        dcav_means_dbeta -= self.cavity_means * self.Ytilde * np.eye(self.beta.shape[0])

        #dcav_vars_dYtilde = 0
        dcav_means_dYtilde = (self.Sigma * self.beta[:, None] / self.diag_Sigma - np.diag(self.beta)) * self.cavity_vars

        dB_dbeta = np.dot(dcav_means_dbeta, dq_dcav_means) + np.dot(dcav_vars_dbeta, dq_dcav_vars)
        dB_dYtilde = np.dot(dcav_means_dYtilde, dq_dcav_means)


        #C TODO
        dC_dYtilde = 0
        dC_dbeta = 0

        #sum gradients from all the different parts
        dL_dbeta = dA_dbeta + dB_dbeta + dC_dbeta
        dL_dYtilde = dA_dYtilde + dB_dYtilde + dC_dYtilde

        dL_dK = np.eye(self.num_data) # TODO

        return np.hstack((dL_dYtilde, dL_dbeta, self.kern.dK_dtheta(dL_dK, self.X)))

    def plot(self):
        pb.errorbar(self.X[:,0],self.Ytilde,yerr=2*np.sqrt(1./self.beta), fmt=None, label='approx. likelihood')
        #pb.errorbar(self.X[:,0]+0.01,self.q_means,yerr=2*np.sqrt(self.q_vars), fmt=None, label='q(f) (non Gauss.)')
        pb.errorbar(self.X[:,0]+0.02,self.mu,yerr=2*np.sqrt(np.diag(self.Sigma)), fmt=None, label='approx. posterior')
        pb.legend()


if __name__=='__main__':
    N = 2
    X = np.random.rand(N)[:,None]
    X = np.sort(X,0)
    Y = np.where(X>0.5,1,0).flatten()
    m = classification(X, Y, GPy.kern.rbf(1, .6, 0.2) + GPy.kern.white(1, 1e-1))
    m.constrain_positive('beta')
    m.constrain_fixed('rbf')
    m.constrain_fixed('white')
    #m.optimize('simplex', max_f_eval=20000, messages=1)
    m.randomize()
    m.checkgrad(verbose=True)

#     def f(b):
#         m['beta'] = b
#         m.log_likelihood()
#         return m.cavity_means[0]
#     def df(b):
#         m['beta'] = b
#         m._log_likelihood_gradients()
#         return m.dcav_means_dbeta[0]
#     mu_grad = GradientChecker(f, df, m.beta, ['beta'])
#     mu_grad.checkgrad(verbose=1)
