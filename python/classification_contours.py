import numpy as np
import pylab as pb
import GPy
from classification2 import classification
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

Y = np.ones(2)
X = np.linspace(-1,1,2)[:,None]*0.5

#build tVB model
k = GPy.kern.rbf(1) + GPy.kern.white(1, 0.01)
m = classification(X,Y,k,link='heaviside')
m.Ytilde = np.ones(2)
m.constrain_fixed('rbf')
m.constrain_fixed('white')
m.randomize()
#m.optimize('bfgs')
m.optimize_restarts()

#build EP model
#m_ep = GPy.models.GPClassification(X,Y, kernel=k)
#m_ep.update_likelihood_approximation()
#m_ep.constrain_fixed('')

resolution = 22
ff_x, ff_y = np.mgrid[-1:3:resolution * 1j,-1:3:resolution * 1j]
ff = np.vstack((ff_x.flatten(), ff_y.flatten())).T
prior = mvn_pdf(ff, k.K(X)).reshape(resolution,resolution)
post = np.where(np.logical_and(ff_x>0, ff_y>0), prior,0)
post /= post.sum()
post /= np.diff(ff_x,axis=0)[0,0]**2


tilted = m.tilted.pdf(ff).prod(1).reshape(resolution, resolution)
tilted /= tilted.sum()
tilted /= np.diff(ff_x,axis=0)[0,0]**2

approx = mvn_pdf(ff, m.Sigma, m.mu).reshape(resolution,resolution)

contours = np.linspace(1e-6,1,20)
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
ax.contour(ff_x, ff_y, approx, contours, colors='r')
ax.contour(ff_x, ff_y, tilted, contours, colors='g')





