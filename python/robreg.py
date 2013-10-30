import numpy as np
import GPy
from varEP import varEP
from quad_tilt import quad_tilt
from functools import partial
from integrate import integrate
import pylab as pb

class robreg(varEP):
    def __init__(self, X, Y, kern=None, in_parallel=False):
        self.Y = Y
        varEP.__init__(self, X, quad_tilt(Y.flatten(), in_parallel=in_parallel) , kern)

    def predict(self, Xnew):
        mu, var = self._predict_raw(Xnew)
        return mu

    def validate(self, Xnew, Ynew):
        """
        return p(Y* | X*) for each of the Xnew, Ynew.
        """
        mu, var = self._predict_raw(Xnew)
        f = partial(integrate, lik=self.tilted.lik, derivs=False)
        quads, numevals = zip(*map(f,Ynew.flatten(), mu.flatten(), np.sqrt(var.flatten())))
        quads = np.vstack(quads)
        return quads[:,0]

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
    import pylab as pb
    pb.ion()
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
