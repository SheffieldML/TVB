# Copyright (c) 2014, James Hensman, Max Zwiessele
# Distributed under the terms of the GNU General public License, see LICENSE.txt

import numpy as np
from scipy.special import gamma, digamma
from scipy import stats

class student_t():
    def __init__(self):
        self._set_params(np.ones(2))
    def _set_params(self, p):
        self.nu, self.lamb = p
        #compute some constants so that they don't appear in a loop
        self._pdf_const = gamma((self.nu + 1)/2.) / gamma(self.nu/2.) * np.sqrt(self.lamb/(self.nu*np.pi) )
        self._dnu_const = 0.5*digamma((self.nu + 1.)/2.) - 0.5*digamma(self.nu/2.) - 0.5/self.nu
    def _get_params(self):
        return np.array([self.nu, self.lamb])
    def _get_param_names(self):
        return ['nu', 'lambda']
    def pdf(self, x, Y):
        x2 = np.square(x-Y)
        return self._pdf_const * np.power(1 + self.lamb*x2/self.nu, -(self.nu + 1.)/2.)
    def dlnpdf_dtheta(self, x, Y):
        x2 = np.square(x-Y)
        dnu = self._dnu_const - 0.5*np.log(1. + self.lamb*x2/self.nu) + 0.5*(self.nu + 1.)*(self.lamb*x2/self.nu**2)/(1. + self.lamb*x2/self.nu)
        dlamb =  0.5/self.lamb - 0.5*(self.nu + 1.)*(x2/self.nu/(1.+self.lamb*x2/self.nu))
        return np.vstack((dnu, dlamb))

    def predictive_values(self, mu, var, percentiles):
        if len(percentiles)==0:
            return mu, []
        samples = (np.random.randn(40e3,*mu.shape) + mu)*np.sqrt(var)
        samples = stats.t.rvs(self.nu, loc=samples, scale=np.array(self.lamb).reshape(1,1))
        qs = [stats.scoreatpercentile(samples,q,axis=0) for q in percentiles]
        return samples.mean(0), qs
