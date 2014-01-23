# Copyright (c) 2014, James Hensman
# Distributed under the terms of the GNU General public License, see LICENSE.txt

import numpy as np
import pylab as pb
import GPy
from classification2 import classification
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

Y = np.ones(2)
#Y[1] = 0
X = np.linspace(-1,1,2)[:,None]*0.4

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

resolution = 100
xmin = m.mu[0] - 3*np.sqrt(m.Sigma[0,0])
xmax = m.mu[0] + 3*np.sqrt(m.Sigma[0,0])
ymin = m.mu[1] - 3*np.sqrt(m.Sigma[1,1])
ymax = m.mu[1] + 3*np.sqrt(m.Sigma[1,1])
ff_x, ff_y = np.mgrid[xmin:xmax:resolution * 1j,ymin:ymax:resolution * 1j]
ff = np.vstack((ff_x.flatten(), ff_y.flatten())).T
prior = mvn_pdf(ff, k.K(X)).reshape(resolution,resolution)
likelihood = ((ff_x>0)==Y[0]) * ((ff_y>0)==Y[1])
post = likelihood*prior
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
#ax.plot([0,3,0,0], [0,0,0,3], post_color, linewidth=2)
#ax.set_xlim(-1,3)
#ax.set_ylim(-1,3)
contours = [0.5]
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
ax.plot(ff_y[0], marg, xmin-1, zdir='x', color=post_color)
marg = post.sum(1)
marg /= marg.sum()
marg /= (ff_y[0][1] - ff_y[0][0])
ax.plot(ff_y[0], marg, ymax+1, zdir='y', color=post_color)

marg = tilted.sum(0)
marg /= marg.sum()
marg /= (ff_y[0][1] - ff_y[0][0])
ax.plot(ff_y[0], marg, xmin-1, zdir='x', color=tilt_color)
marg = tilted.sum(1)
marg /= marg.sum()
marg /= (ff_y[0][1] - ff_y[0][0])
ax.plot(ff_y[0], marg, ymax + 1, zdir='y', color=tilt_color)

marg = approx.sum(0)
marg /= marg.sum()
marg /= (ff_y[0][1] - ff_y[0][0])
ax.plot(ff_y[0], marg, xmin-1, zdir='x', color=approx_color)
marg = approx.sum(1)
marg /= marg.sum()
marg /= (ff_y[0][1] - ff_y[0][0])
ax.plot(ff_y[0], marg, ymax + 1, zdir='y', color=approx_color)

contours = np.linspace(0, approx.max(),20)[1:]
ax.contour(ff_x, ff_y, np.clip(approx,0,contours.max()), contours, colors=approx_color)
contours = np.linspace(0,tilted.max(),20)[1:]
ax.contour(ff_x, ff_y, tilted, contours, colors=tilt_color)

ax.set_zlim(0,1)
ax.set_xlim(xmin -1, xmax)
ax.set_ylim(ymin , ymax+1)
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






