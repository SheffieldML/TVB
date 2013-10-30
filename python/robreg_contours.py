import numpy as np
import pylab as pb
import GPy
from robreg import robreg
from mpl_toolkits.mplot3d import Axes3D
pb.close('all')

def mvn_pdf(X, sigma, mu=None):
    N,D = X.shape
    if mu is None:
        mu = np.zeros(D)
    Si, L, Li, logdet = GPy.util.linalg.pdinv(sigma)
    X_ = X-mu
    mah = np.sum(np.dot(X_,Si)*X_,1)
    return np.exp(-0.5*np.log(2*np.pi) - 0.5*logdet -0.5*mah)

Y = np.array([-1,1])
X = np.linspace(-1,1,2)[:,None]*0.1

#build tVB model
k = GPy.kern.rbf(1) + GPy.kern.white(1, 0.01)
m = robreg(X,Y,k)
m.Ytilde = np.ones(2)
m.constrain_fixed('rbf')
m.constrain_fixed('white')
m['lam'] = 20
m.constrain_fixed('(nu|lam)')

m.randomize()
m.optimize_restarts(optimizer='bfgs', robust=1)

resolution = 200
ff_x, ff_y = np.mgrid[-3:3:resolution * 1j,-3:3:resolution * 1j]
ff = np.vstack((ff_x.flatten(), ff_y.flatten())).T
prior = mvn_pdf(ff, k.K(X)).reshape(resolution,resolution)
likelihood = m.tilted.lik.pdf(ff_x, Y[0]) *m.tilted.lik.pdf(ff_y, Y[1])
post = prior*likelihood
post /= post.sum()
post /= np.diff(ff_x,axis=0)[0,0]**2

pb.subplot(131)
pb.contour(ff_x, ff_y, prior)
pb.subplot(132)
pb.contour(ff_x, ff_y, likelihood)
pb.subplot(133)
pb.contour(ff_x, ff_y, post)

tilted = m.tilted.pdf(ff).prod(1).reshape(resolution, resolution)
tilted /= tilted.sum()
tilted /= np.diff(ff_x,axis=0)[0,0]**2

approx = mvn_pdf(ff, m.Sigma, m.mu).reshape(resolution,resolution)

pb.figure()
contours = np.linspace(0,post.max(),20)[1:]
pb.contour(ff_x, ff_y, post, contours, colors='k')
pb.contour(ff_x, ff_y, approx, contours, colors='r')
pb.contour(ff_x, ff_y, tilted, contours, colors='g')

f = pb.figure()
ax = f.add_subplot(111, projection='3d')
ax.plot_wireframe(ff_x, ff_y, post, color='k')
ax.plot_wireframe(ff_x, ff_y, approx, color='r')
ax.plot_wireframe(ff_x, ff_y, tilted, color='g')

f = pb.figure()
ax = f.add_subplot(111, projection='3d')
ax.contour(ff_x, ff_y, post,contours, colors='k')
ax.contour(ff_x, ff_y, np.clip(approx, 0, post.max()), contours, colors='r')
ax.contour(ff_x, ff_y, tilted, contours, colors='g')





