import numpy as np
import pylab as pb
import GPy
from robreg import robreg
pb.close('all')

data = GPy.util.datasets.xw_pen()
X = data['X']
Y = data['Y']
Xtest = np.linspace(175, 275, 200)[:,None]

#use a subset of the data for speed
X = X[175:275]
Y = Y[175:275]


#build and optimize dumb GP regression
m_gauss = GPy.models.GPRegression(X, Y, kernel=GPy.kern.rbf(1) + GPy.kern.bias(1))
m_gauss.optimize('bfgs')
Ypred, var_pred, _, _ = m_gauss.predict(Xtest)
pb.figure()
pb.plot(Ypred[:,1], Ypred[:,0])


#build and optimize a model (separate models for each output right now...
#m1 = robreg(X, Y[:,0], kern = GPy.kern.rbf(1) + GPy.kern.white(1, 0.01) + GPy.kern.bias(1))
#kern = GPy.kern.periodic_Matern32(1, variance=100.,lengthscale=10., period=40, lower=140, upper=300) + GPy.kern.white(1, 1.) + GPy.kern.bias(1, 100.)
kern = GPy.kern.Matern32(1, variance=10.,lengthscale=10.) + GPy.kern.white(1, 1.) + GPy.kern.bias(1, 100.)
m1 = robreg(X, Y[:,0], kern = kern.copy())
#m1.constrain_positive('.*period')
m1.Ytilde = m1.Y.copy()
m1['beta'] = 1.
m1.constrain_positive('lambda')
m1.constrain_fixed('nu', 4.)
m1.optimize('bfgs', messages=1)
stop

Ypred_1 = m1.predict(Xtest)
m2 = robreg(X, Y[:,1], kern = kern.copy())
m2.Ytilde = m2.Y.copy()
m2['beta'] = 1.
m2.constrain_positive('lambda')
#m2.constrain_positive('.*period')
m2.constrain_fixed('nu', 4.)
m2.optimize('bfgs', messages=1)
m2.plot()
Ypred_2 = m2.predict(Xtest)

pb.figure()
pb.plot(Ypred_1, Ypred_2)


