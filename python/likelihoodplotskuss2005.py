'''
Created on 12 Sep 2013

@author: maxz, James Hensman
'''
import numpy as np
from classification1 import classification as class1
from classification2 import classification as class2
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
    N = 10
    X = X[:N]
    Y = Y[:N]

    gridsize = 6  # x gridsipyze
    l, a = np.meshgrid(np.linspace(0, 5, gridsize), np.linspace(0, 6, gridsize))
    kern = GPy.kern.rbf(X.shape[1]) + GPy.kern.white(X.shape[1])

    #set up the tVB model
    m = class2(X, Y, kern)
    m.tilted.do_entropy = True
    m.no_K_grads_please = True

    #set up the EP model
    link = GPy.likelihoods.noise_models.gp_transformations.Heaviside()
    #link = GPy.likelihoods.noise_models.gp_transformations.Probit()
    lik = GPy.likelihoods.binomial(link)
    m_ep = GPy.models.GPClassification(X, Y.reshape(-1,1), kernel=kern.copy())

    #loop!
    Z_tVB = np.zeros((gridsize, gridsize))
    Z_tVB_alt = np.zeros((gridsize, gridsize))
    Z_EP = np.zeros((gridsize, gridsize))
    for i in range(gridsize):
        for j in range(gridsize):
            aa = a[i,j]
            ll = l[i,j]
            print "Doing point: {:.2} {:.2}".format(ll, aa)
            #do the tVB model first
            m.constrain_fixed('rbf_len', np.exp(ll))
            m.constrain_fixed('rbf_var', np.exp(aa))
            m.constrain_fixed('white', 1.)
            m.randomize()
            m.optimize('bfgs', messages=0)#, bfgs_factor=1e20)
            Z_tVB[i,j] =  m.log_likelihood()
            #Z_tVB_alt[i,j] = m.alternative_log_likelihood()

            #Do EP
            m_ep._set_params(np.array([np.exp(aa), np.exp(ll), 1e-6]))
            m_ep.update_likelihood_approximation()
            Z_EP[i,j] = m_ep.log_likelihood()

    pb.figure('tVB')
    c = pb.contour(l,a,Z_tVB, 10, color='k')
    pb.clabel(c)

    pb.figure('tVB_alt')
    c = pb.contour(l,a,Z_tVB_alt, 10, color='k')
    pb.clabel(c)

    pb.figure('EP')
    c = pb.contour(l,a,Z_EP, 10, color='k')
    pb.clabel(c)

