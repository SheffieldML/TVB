import numpy as np
import pylab as pb
pb.ion()
import GPy
from truncnorm import truncnorm
from GPy.models.gradient_checker import GradientChecker
from scipy.special import erf
from varEP import varEP
from quad_tilt import quad_tilt

class robreg(varEP):
    def __init__(self, X, Y, kern=None):
        self.Y = Y
        varEP.__init__(self, X, quad_tilt(Y.flatten()) , kern)

    def predict(self, Xnew):
        mu, var = self._predict_raw(Xnew)
        return mu

    def plot(self):
        if self.X.shape[1]==1:
            pb.figure()
            Xtest, xmin, xmax = GPy.util.plot.x_frame1D(self.X)
            mu = self.predict(Xtest)
            pb.plot(Xtest.flatten(), mu.flatten(), 'b')
            pb.plot(self.X, self.Y, 'kx', mew=1)
        elif self.X.shape[1]==2:
            pb.figure()
            Xtest,xx,yy, xymin, xymax = GPy.util.plot.x_frame2D(self.X)
            p = self.predict(Xtest)
            c = pb.contour(xx,yy,p.reshape(*xx.shape), colors='k')
            pb.clabel(c)


if __name__=='__main__':
    pb.close('all')
    #construct a data set
    X = np.linspace(0,1,20)[:,None]
    Y = np.sin(2*np.pi*X) + np.random.randn(*X.shape)*0.05
    Y[14] += 2
    #build and optimize a model
    m = robreg(X, Y)
    m.constrain_positive('(nu|lambda)')
    #m.randomize();     m.checkgrad('(nu|lamb)',verbose=True)
    m.checkgrad('(nu|lamb)',verbose=True)
    #m.randomize()
    #m.optimize('bfgs')
    #m.plot()
