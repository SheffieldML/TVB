'''
Created on 12 Sep 2013

@author: maxz
'''
import numpy
from classification1 import classification
from GPy.kern.constructors import rbf, white
from GPy.models.gp_classification import GPClassification
import sys

if __name__ == '__main__':
    # read in the data:
    data = numpy.loadtxt("ionosphere.dat", str, delimiter=',')
    X = numpy.array(data[:, 2:34], dtype=float)
    X = X - X.mean(0); X = X / X.std(0)
    labels = data[:, -1]
    Y = numpy.where(labels == 'b', 0, 1)[:, None]

    gridsize = 24  # x gridsipyze
    l, a = numpy.meshgrid(numpy.linspace(-1, 5, gridsize), numpy.linspace(-1, 5, gridsize))
    kern = rbf(X.shape[1]) + white(X.shape[1], 1e-5)

    mm = GPClassification(X, Y, kernel=kern)
    m = classification(X, Y, kern)
    m.constrain_positive('beta')
    m.constrain_fixed('rbf')
    m.constrain_fixed('white')

    def set_param_and_evaluate(l, a):
        # m.kern['rbf_len'] = numpy.exp(l)
        # m.kern['rbf_var'] = numpy.exp(a)
        # m.update_likelihood_approximation()
        m.constrain_fixed('rbf_len', numpy.exp(l))
        m.constrain_fixed('rbf_var', numpy.exp(a))
        print "Optimizing: {:.2} {:.2}".format(l, a)
        m.optimize('bfgs', messages=1, gtol=1)
        return m.log_likelihood()

    surface_func = numpy.vectorize(set_param_and_evaluate)
    Z = surface_func(l, a)
