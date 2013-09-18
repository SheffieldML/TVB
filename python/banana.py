import numpy as np
import pylab as pb
import GPy
from classification1 import classification

d = np.loadtxt('../data/banana-5-1tra.dat', skiprows=7, delimiter=',')
d = d[:300]
X = d[:,:2]
Y = np.where(d[:,2]==1,1,0)
k = GPy.kern.rbf(2) + GPy.kern.white(2)

m = classification(X,Y.flatten(),k)
m.randomize()#
m.optimize('bfgs', max_iters=20, messages=1)
m.plot()
