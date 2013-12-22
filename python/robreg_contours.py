import numpy as np
import pylab as pb
import GPy
from robreg import robreg
from mpl_toolkits.mplot3d import Axes3D
pb.close('all')

post_color = 'k'
tilt_color='#0172B2'
approx_color= '#CC6600'

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
m.optimize('bfgs')
m.optimize_restarts(optimizer='bfgs', robust=1)

resolution = 200
ff_x, ff_y = np.mgrid[-1.5:1.5:resolution * 1j,-1.5:1.5:resolution * 1j]
ff = np.vstack((ff_x.flatten(), ff_y.flatten())).T
prior = mvn_pdf(ff, k.K(X)).reshape(resolution,resolution)
likelihood = m.tilted.lik.pdf(ff_x, Y[0]) *m.tilted.lik.pdf(ff_y, Y[1])
post = prior*likelihood
post /= post.sum()
post /= np.diff(ff_x,axis=0)[0,0]**2

tilted = m.tilted.pdf(ff).prod(1).reshape(resolution, resolution)
tilted /= tilted.sum()
tilted /= np.diff(ff_x,axis=0)[0,0]**2

approx = mvn_pdf(ff, m.Sigma, m.mu).reshape(resolution,resolution)
approx /= approx.sum()
approx /= np.diff(ff_x,axis=0)[0,0]**2


pb.figure(figsize=(8,10))
ax = pb.subplot2grid((4,3),(0,0), colspan=1, rowspan=1)
contours = np.linspace(0,prior.max(),10)[1:]
ax.contour(ff_x, ff_y, prior, contours, colors=post_color)
ax.set_xticks([]);ax.set_yticks([])
ax = pb.subplot2grid((4,3),(0,1))
contours = np.linspace(0,likelihood.max(),10)[1:]
ax.contour(ff_x, ff_y, likelihood, contours, colors=post_color)
ax.set_xticks([]);ax.set_yticks([])
ax = pb.subplot2grid((4,3),(0,2))
contours = np.linspace(0,post.max(),10)[1:]
ax.contour(ff_x, ff_y, post, contours, colors=post_color)
ax.set_xticks([]);ax.set_yticks([])

ax = pb.subplot2grid((4,3),(1,0), colspan=3, rowspan=3, projection='3d')
contours = np.linspace(0,post.max(),20)[1:]
ax.contour(ff_x, ff_y, post,contours, colors=post_color)

marg = post.sum(0)
marg /= marg.sum()
marg /= (ff_y[0][1] - ff_y[0][0])
ax.plot(ff_y[0], marg, -2, zdir='x', color=post_color)
marg = post.sum(1)
marg /= marg.sum()
marg /= (ff_y[0][1] - ff_y[0][0])
ax.plot(ff_y[0], marg, 2, zdir='y', color=post_color)

marg = tilted.sum(0)
marg /= marg.sum()
marg /= (ff_y[0][1] - ff_y[0][0])
ax.plot(ff_y[0], marg, -2, zdir='x', color=tilt_color)
marg = tilted.sum(1)
marg /= marg.sum()
marg /= (ff_y[0][1] - ff_y[0][0])
ax.plot(ff_y[0], marg, 2, zdir='y', color=tilt_color)

marg = approx.sum(0)
marg /= marg.sum()
marg /= (ff_y[0][1] - ff_y[0][0])
marg = np.clip(marg,0,4)
ax.plot(ff_y[0], marg, -2, zdir='x', color=approx_color)
marg = approx.sum(1)
marg /= marg.sum()
marg /= (ff_y[0][1] - ff_y[0][0])
marg = np.clip(marg,0,4)
ax.plot(ff_y[0], marg, 2, zdir='y', color=approx_color)

contours = np.linspace(0,max(tilted.max(), post.max()),20)[1:]
ax.contour(ff_x, ff_y, np.clip(approx,0,contours.max()), contours, colors=approx_color)
contours = np.linspace(0,tilted.max(),20)[1:]
ax.contour(ff_x, ff_y, tilted, contours, colors=tilt_color)

ax.set_zlim(0,2)
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
pb.tight_layout()
pb.draw()


# and a boring contour
pb.figure()
contours = np.linspace(0,post.max(),20)[1:]
pb.contour(ff_x, ff_y, post, contours, colors=post_color)
contours = np.linspace(0,approx.max(),20)[1:]
pb.contour(ff_x, ff_y, approx, contours, colors=approx_color)
contours = np.linspace(0,tilted.max(),20)[1:]
pb.contour(ff_x, ff_y, tilted, contours, colors=tilt_color)



