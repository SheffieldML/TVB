'''
Created on 12 Sep 2013

@author: maxz, James Hensman
'''
import numpy as np
from classification1 import classification
import GPy
import sys
import pylab as pb
pb.close('all')

if __name__ == '__main__':
    # read in the data:
    data = np.loadtxt("ionosphere.dat", str, delimiter=',')
    X = np.array(data[:, 2:34], dtype=float)
    X = X - X.mean(0); X = X / X.std(0)
    labels = data[:, -1]
    Y = np.where(labels == 'b', 0, 1)[:, None].flatten()

    #cut some data out ( as kuss)
    X = X[:200]
    Y = Y[:200]

    gridsize = 10  # x gridsipyze
    l, a = np.meshgrid(np.linspace(0, 5, gridsize), np.linspace(0, 5, gridsize))
    kern = GPy.kern.rbf(X.shape[1]) + GPy.kern.white(X.shape[1])


    m = classification(X, Y, kern)
    m.no_K_grads_please = True
    def set_param_and_evaluate(l, a):
        # m.kern['rbf_len'] = np.exp(l)
        # m.kern['rbf_var'] = np.exp(a)
        m.constrain_fixed('rbf_len', np.exp(l))
        m.constrain_fixed('rbf_var', np.exp(a))
        m.constrain_fixed('white', 1.)
        m.randomize()
        print "Optimizing: {:.2} {:.2}".format(l, a)
        m.optimize('bfgs', messages=0)#, bfgs_factor=1e20)
        return m.alternative_log_likelihood()
    surface_func = np.vectorize(set_param_and_evaluate)
    Z = surface_func(l, a)

    #link = GPy.likelihoods.noise_models.gp_transformations.Heaviside()
    link = GPy.likelihoods.noise_models.gp_transformations.Probit()
    lik = GPy.likelihoods.binomial(link)
    #m_ep = GPy.models.GPClassification(X,likelihood=GPy.likelihoods.EP(Y.reshape(-1,1), lik), kernel = m.kern.copy())
    m_ep = GPy.models.GPClassification(X,Y.reshape(-1,1), kernel = m.kern.copy())
    def set_param_and_evaluate_ep(l, a):
        m_ep._set_params(np.array([np.exp(a), np.exp(l), 1e-6]))
        m_ep.update_likelihood_approximation()
        print "EP-ing: {:.2} {:.2}".format(l, a)
        return m_ep.log_likelihood()
    surface_func_ep = np.vectorize(set_param_and_evaluate_ep)
    Z_ep = surface_func_ep(l, a)

    c = pb.contour(l,a,Z, 10)
    pb.clabel(c)
    pb.colorbar()
    pb.figure()
    pb.contour(l,a,np.exp(Z))
    pb.colorbar()

    pb.figure()
    c = pb.contour(l,a,Z_ep, 10)
    pb.clabel(c)
    pb.colorbar()
    pb.figure()
    pb.contour(l,a,np.exp(Z_ep))
    pb.colorbar()

