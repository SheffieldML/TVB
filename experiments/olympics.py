import numpy as np
import GPy
from robreg import robreg

data = GPy.util.datasets.olympic_marathon_men()
X = data['X']
Y = data['Y']

#build and optimize dumb GP regression
m = GPy.models.GPRegression(X, Y, kernel = GPy.kern.rbf(1) + GPy.kern.bias(1))
m.optimize('bfgs')
m.plot()

#build and optimize a model
m = robreg(X, Y.flatten(), kern = GPy.kern.rbf(1) + GPy.kern.white(1, 0.001) + GPy.kern.bias(1))
m.constrain_positive('lambda')
m.constrain_positive('nu')
m.constrain_fixed('nu', 3)
m.optimize('bfgs')
m.plot()
