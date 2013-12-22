import numpy as np
import pylab as pb
pb.ion()
import GPy
from truncnorm import truncnorm
from scipy.special import erf
import tilted
from varEP import varEP
from varEP2 import varEP as varEP2

class classification(varEP2):
    def __init__(self, X, Y, kern=None, link='probit'):
        self.Y = Y
        if link=='probit':
            varEP2.__init__(self,X, tilted.Probit(Y.flatten()), kern)
        elif link=='heaviside':
            varEP2.__init__(self,X, tilted.Heaviside(Y.flatten()), kern)
        else:
            raise ValueError('bad link name')

    def predict(self, Xnew):
        mu, var = self._predict_raw(Xnew)
        return self.tilted.predict(mu, var)

    def plot(self):
        if self.X.shape[1]==1:
            pb.figure()
            Xtest, xmin, xmax = GPy.util.plot.x_frame1D(self.X)
            mu, var = self._predict_raw(Xtest)
            pb.plot(self.X, self.Y, 'kx', mew=1)
            pb.plot(Xtest, 0.5*(1+erf(mu/np.sqrt(2.*var))), linewidth=2)
            pb.ylim(-.1, 1.1)
        elif self.X.shape[1]==2:
            pb.figure()
            Xtest,xx,yy, xymin, xymax = GPy.util.plot.x_frame2D(self.X)
            p = self.predict(Xtest)
            c = pb.contour(xx,yy,p.reshape(*xx.shape), [0.1, 0.25, 0.5, 0.75, 0.9], colors='k')
            pb.clabel(c)
            i1 = self.Y==1
            pb.plot(self.X[:,0][i1], self.X[:,1][i1], 'rx', mew=2, ms=8)
            i2 = self.Y==0
            pb.plot(self.X[:,0][i2], self.X[:,1][i2], 'wo', mew=2, mec='b')


if __name__=='__main__':
    pb.close('all')
    #construct a data set
    N = 20
    X = np.random.rand(N)[:,None]
    X = np.sort(X,0)
    Y = np.zeros(N)
    Y[X[:, 0] < 3. / 4] = 1.
    Y[X[:, 0] < 1. / 4] = 0.

    #build and optimize a model
    m = classification(X, Y)
    m.randomize();     m.checkgrad(verbose=True)
    #m.randomize()
    #m.optimize('bfgs')
    #m.plot()
