import numpy as np
import pylab as pb
pb.ion()
pb.close('all')
import GPy
from classification2 import classification
np.random.seed(0)

d = np.loadtxt('../data/banana-5-1tra.dat', skiprows=7, delimiter=',')
d = d[:300]
X = d[:,:2]
Y = np.where(d[:,2]==1,1,0)
k = GPy.kern.rbf(2) + GPy.kern.white(2)

#build our model
m = classification(X,Y.flatten(),k)
m.randomize()#
m.optimize('bfgs', max_iters=500, messages=1)
m.plot()
pb.title('variational EP')

#build an EP model
link = GPy.likelihoods.noise_models.gp_transformations.Heaviside()
lik = GPy.likelihoods.bernoulli(link)
l = GPy.likelihoods.EP(Y.reshape(-1,1),lik)
m_ep1 = GPy.models.GPClassification(X,likelihood=l, kernel = m.kern.copy())
m_ep1.update_likelihood_approximation()
m_ep1.plot(levels=[0.1, 0.25, 0.5, 0.75, 0.9])
pb.title('EP (same kern params)')

#now optimize against the ep approximation to the marg. lik.
m_ep2 = m_ep1.copy()
m_ep2.pseudo_EM()
m_ep2.plot(levels=[0.1, 0.25, 0.5, 0.75, 0.9])
pb.title('EP (pseudo-EM)')
