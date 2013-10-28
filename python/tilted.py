import numpy as np
import pylab as pb
from scipy.special import erf
from GPy.models.gradient_checker import GradientChecker

class Tilted(object):
    def __init__(self, Y):
        self.Y = Y
    def set_cavity(self, mu, sigma2):
        self.mu, self.sigma2 = mu, sigma2
        self.sigma = np.sqrt(self.sigma2)

def norm_cdf(x):
    return 0.5*(1+erf(x/np.sqrt(2.)))
def norm_pdf(x):
    return np.exp(-0.5*np.square(x))/np.sqrt(2.*np.pi)


class Heaviside(Tilted):
    def __init__(self, Y, do_entropy=False):
        Tilted.__init__(self,Y)
        self.Ysign = np.where(self.Y==1,1,-1)
        self.do_entropy = do_entropy
    def _set_params(self, x):
        pass
    def _get_params(self):
        return np.zeros(0)
    def _get_param_names(self):
        return []

    def set_cavity(self, mu, sigma2):
        Tilted.set_cavity(self, mu, sigma2)
        self.a = self.Ysign*self.mu/self.sigma
        self.Z = norm_cdf(self.a)
        self.N = norm_pdf(self.a)
        self.N_Z = self.N/self.Z
        self.N_Z2 = np.square(self.N_Z)
        self.N_Z3 = self.N_Z2*self.N_Z

        #compute moments
        self.mean = self.mu + self.Ysign*self.sigma*self.N_Z
        self.var = self.sigma2*(1. - self.a * self.N_Z - self.N_Z2)

        #derivatives of moments
        self.dmean_dmu = 1. - (self.N_Z2 + self.a * self.N_Z)
        self.dmean_dsigma2 = self.Ysign * self.N_Z * 0.5/self.sigma * (1 + self.a * (self.a + self.N_Z))
        self.dvar_dmu = -self.Ysign*self.sigma *self.N_Z + self.a * self.mu * self.N_Z + 3 * self.mu * self.N_Z2 + 2 * self.Ysign * self.sigma * self.N_Z3
        self.dvar_dsigma2 = 1 - self.N_Z * (self.N_Z + self.a * (.5 + .5*self.a**2 + self.N_Z * (1.5*self.a + self.N_Z)))

        #derivatives of Z
        self.dZ_dmu = self.N*self.Ysign/self.sigma
        self.dZ_dsigma2 = -0.5*self.N*self.Ysign*self.mu/self.sigma2/self.sigma

        if self.do_entropy:
            #compute entropy
            self.H = 0.5*np.log(2*np.pi*self.sigma2) + np.log(self.Z) + 0.5*(self.mu**2 + self.var + self.mean**2 - 2*self.mean*self.mu)/self.sigma2

            #entropy derivatives
            self.dH_dmu = self.Ysign*(0.5 / self.sigma2) * (self.N_Z * (self.sigma + self.mu ** 2 / self.sigma) + self.Ysign*self.mu * self.N_Z2)
            self.dH_dsigma2 =(
                    1./(2*self.sigma2) - self.N_Z * (self.a/(2*self.sigma2))
                    + .5 * (1./(self.sigma2)) * (self.dvar_dsigma2 + (2*self.mean - 2*self.mu) * self.dmean_dsigma2)
                    - 0.5/(self.sigma2**2) * (self.mu**2 + self.var + self.mean**2 - 2*self.mean*self.mu)
                    )

class Probit(Tilted):
    def __init__(self, Y):
        super(Probit, self).__init__(Y)
        self.Ysign = np.where(self.Y==1,1,-1)

    def set_cavity(self, mu, sigma2):
        Tilted.set_cavity(self, mu, sigma2)
        
        sigma2p1 = 1 + self.sigma2
        da_dsigma2 = -.5*self.Ysign*self.mu*np.power(sigma2p1,-3./2)
        
        self.a = self.Ysign*self.mu/(np.sqrt(sigma2p1))
        
        self.Z = norm_cdf(self.a)
        self.N = norm_pdf(self.a)
        self.N_Z = self.N/self.Z
        self.N_Z2 = np.square(self.N_Z)
        self.N_Z3 = self.N_Z2*self.N_Z
        
        self.mean = self.mu + self.Ysign*self.sigma2*self.N_Z/(np.sqrt(sigma2p1))
        self.var = self.sigma2*(1. - ((self.sigma2 * self.N_Z / sigma2p1) * (self.a + self.N_Z)))

        self.dZ_dmu = self.N*self.Ysign/np.sqrt(sigma2p1)
        self.dZ_dsigma2 = -0.5*self.N*self.Ysign*self.mu*np.power(sigma2p1, -1.5)


        self.dmean_dmu = (1 - self.sigma2/sigma2p1 * self.N_Z * (self.a + self.N_Z))
        
        self.dN_dsigma2 = - self.N * self.a * da_dsigma2
        self.dmean_dsigma2 = (self.Ysign*self.N_Z/np.sqrt(sigma2p1)
                           *(1+self.sigma2*(
                                self.N_Z*self.Ysign*self.mu/(2*np.sqrt(sigma2p1))
                                +self.Ysign*self.a*self.mu/(self.Z*np.sqrt(sigma2p1))
                                -.5)
                             )/sigma2p1) 
        
        self.dvar_dmu = -((self.Ysign/np.sqrt(sigma2p1)) * (np.square(self.sigma2)/sigma2p1) * self.N_Z
                         #* ((self.a + (self.Ysign/sigma2p1) + 2) * (self.N_Z + self.a)))
                          * (1 + (self.a + 2*self.N_Z) * (self.N_Z + self.a)))

        dN_Z_dmu = -self.Ysign/np.sqrt(sigma2p1)*(self.a*self.N_Z + self.N_Z2)
        self.dvar_dmu = -self.sigma2**2/sigma2p1*(dN_Z_dmu*self.a + 2.*dN_Z_dmu*self.N_Z + self.N_Z*self.Ysign/np.sqrt(sigma2p1))

        self.dvar_dsigma2 = self.var/self.sigma2\
                + (self.N_Z*self.a + self.N_Z2)/np.square(sigma2p1)\
                + self.sigma2/sigma2p1*(-0.5*self.Y_sign*self.mu*self.N_Z*np.power(sigma2p1, -1.5)\
                    + self.a*dN_Z_dsigma2 + 2.*self.N_Z*dN_Z_dsigma2)


if __name__=='__main__':
    N = 4
    Y = np.random.randint(2,size=N)
    Y[Y==0] = -1
    probit = Probit(Y)
    mu = np.random.randn(N)
    sigma2 = np.random.rand(N)
    
    #gradcheck for Z wrt mu
    def f(mu):
        probit.set_cavity(mu, sigma2)
        return probit.Z
    def df(mu):
        probit.set_cavity(mu, sigma2)
        return probit.dZ_dmu
    m = GradientChecker(f,df,np.random.randn(N))
    m.checkgrad(verbose=1)

    #gradcheck for Z wrt sigma2
    def f(sigma2):
        probit.set_cavity(mu, sigma2)
        return probit.Z
    def df(sigma2):
        probit.set_cavity(mu, sigma2)
        return probit.dZ_dsigma2
    m = GradientChecker(f,df,np.random.rand(N))
    m.checkgrad(verbose=1)

    #gradcheck for mean wrt mu
    def f(mu):
        probit.set_cavity(mu, sigma2)
        return probit.mean
    def df(mu):
        probit.set_cavity(mu, sigma2)
        return probit.dmean_dmu
    m = GradientChecker(f,df,np.random.randn(N))
    m.checkgrad(verbose=1)

    #gradcheck for var wrt mu
    def f(mu):
        probit.set_cavity(mu, sigma2)
        return probit.var
    def df(mu):
        probit.set_cavity(mu, sigma2)
        return probit.dvar_dmu
    m = GradientChecker(f,df,np.random.randn(N))
    m.checkgrad(verbose=1)

	£ gradcheck dN_dsigma2    
	def f(sigma2):
        probit.set_cavity(mu, sigma2)
        return probit.N
    def df(sigma2):
        probit.set_cavity(mu, sigma2)
        return probit.dN_dsigma2
    m = GradientChecker(f,df,np.random.rand(N))
    print 'dN_dsigma2'  
    m.checkgrad(verbose=1)

    #gradcheck for mean wrt sigma2
	def f(sigma2):
        probit.set_cavity(mu, sigma2)
        return probit.mean
    def df(sigma2):
        probit.set_cavity(mu, sigma2)
        return probit.dmean_dsigma2
    m = GradientChecker(f,df,np.random.rand(N))    
    m.checkgrad(verbose=1)

    #gradcheck for var wrt sigma2
    def f(sigma2):
        probit.set_cavity(mu, sigma2)
        return probit.var
    def df(sigma2):
        probit.set_cavity(mu, sigma2)
        return probit.dvar_dsigma2
    m = GradientChecker(f,df,np.random.rand(N))
    m.checkgrad(verbose=1)


#     from truncnorm import truncnorm
#     mu = np.random.randn(2)
#     sigma2 = np.exp(np.random.randn(2))
#     Y = np.array([1,0])
#     tns = [truncnorm(mu[0], sigma2[0], 'left'), truncnorm(mu[1], sigma2[1], 'right')]
#     tilted = Heaviside(Y)
#     tilted.set_cavity(mu, sigma2)


