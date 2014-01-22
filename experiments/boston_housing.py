import numpy as np
import GPy
from robreg import robreg
from integrate import integrate
from sklearn import cross_validation as cv
from scipy import stats

data = GPy.util.datasets.boston_housing()
X = data['X']
Y = data['Y']

X -= X.mean(0)
X /= X.std()
Y -= Y.mean(0)
Y /= Y.std()


def compare(itest, itrain, dof='opt'):
    kern = GPy.kern.rbf(13, ARD=True) + GPy.kern.white(13, 0.001)

    #build and optimize dumb GP regression
    m_gaus = GPy.models.GPRegression(X[itrain], Y[itrain], kernel=kern.copy())
    m_gaus.optimize('bfgs')
    Ypred, var_pred, _, _ = m_gaus.predict(X[itest])
    print 'Gaussian MAE:', np.mean(np.abs(Ypred - Y[itest]))
    log_prob = -0.5*np.log(2*np.pi) - 0.5*np.log(var_pred) - 0.5*np.square(Y[itest] - Ypred)/var_pred
    print 'Gaussian M neg_log_prob:', -np.mean(log_prob)
    print stats.scoreatpercentile(log_prob,5), stats.scoreatpercentile(log_prob,95)

    #build and optimize a model
    m = robreg(X[itrain], Y[itrain], kern=kern.copy(), in_parallel=True)
    m.Ytilde = Y[itrain].copy().flatten()
    m.beta = np.ones_like(m.beta)
    m._set_params(m._get_params())
    m.constrain_positive('lambda')
    if not dof is 'opt':
        m.constrain_fixed('nu', dof)
    else:
        m.constrain_bounded('nu', 1. ,6.)
    m.optimize('bfgs')
    Ypred = m.predict(X[itest]).reshape(-1,1)

    print 'student-t MAE:', np.mean(np.abs(Ypred - Y[itest]))
    log_prob = np.log(m.validate(X[itest], Y[itest]))
    print 'student-t neg_log_prob:', -np.mean(log_prob)
    print stats.scoreatpercentile(log_prob,5), stats.scoreatpercentile(log_prob,95)

    if dof is 'opt':
        dof = m['nu']

    nm = GPy.likelihoods.student_t(deg_free=dof, sigma2=1.)
    lik = GPy.likelihoods.Laplace(Y[itrain].reshape(-1,1), nm)
    m_laplace = GPy.models.GPRegression(X[itrain], Y=None, likelihood=lik, kernel=kern.copy())
    m_laplace.constrain_positive('.*noise')
    m_laplace.optimize('bfgs')
    Ypred = m_laplace.predict(X[itest])[0]
    print 'student-t Laplace MAE:', np.mean(np.abs(Ypred - Y[itest]))
    log_prob = m_laplace.log_predictive_density(X[itest], Y[itest].reshape(-1,1))
    print 'student-t Laplace neg_log_prob:', -np.mean(log_prob)

    print

    return m_gaus, m, m_laplace


#for itrain, itest in cv.KFold(X.shape[0], 2):

#compare(slice(0,10), slice(10,None), 2.)
#m1, m2, m3 = compare(slice(0,100), slice(100,None), 3.)
m1, m2, m3 = compare(slice(0,100), slice(100,None), 'opt')
#compare(slice(0,10), slice(10,None), 4.)
#compare(slice(0,10), slice(10,None), 5.)
