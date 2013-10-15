import numpy as np
import GPy
from robreg import robreg
from integrate import integrate
from sklearn import cross_validation as cv

data = GPy.util.datasets.boston_housing()
X = data['X']
Y = data['Y']

X -= X.mean(0)
X /= X.std()
Y -= Y.mean(0)
Y /= Y.std()

for itrain, itest in cv.KFold(X.shape[0], 10):

    #build and optimize dumb GP regression
    m = GPy.models.GPRegression(X[itrain], Y[itrain], kernel=GPy.kern.rbf(13, ARD=True))
    m.optimize('bfgs')
    Ypred, var_pred, _, _ = m.predict(X[itest])
    print 'Gaussian MAE:', np.mean(np.abs(Ypred - Y[itest]))
    print 'Gaussian M neg_log_prob:', -np.mean(-0.5*np.log(2*np.pi) - 0.5*np.log(var_pred) - 0.5*np.square(Y[itest] - Ypred)/var_pred)

    #build and optimize a model
    m = robreg(X[itrain], Y[itrain], kern = GPy.kern.rbf(13, ARD=True) + GPy.kern.white(13, 0.001))
    m.constrain_positive('lambda')
    m.constrain_fixed('nu', 4.)
    m.optimize('bfgs')
    Ypred = m.predict(X[itest]).reshape(-1,1)

    print 'student-t MAE:', np.mean(np.abs(Ypred - Y[itest]))
    print 'student-t neg_log_prob:', -np.mean(np.log(m.validate(X[itest], Y[itest])))
    print
