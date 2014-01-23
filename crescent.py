# Copyright (c) 2014, James Hensman
# Distributed under the terms of the GNU General public License, see LICENSE.txt

import numpy as np
import pylab as pb
import GPy
from classification1 import classification

d = GPy.util.datasets.crescent_data()
X = d['X']
Y = np.where(d['Y']==1,1,0)
k = GPy.kern.rbf(2) + GPy.kern.white(2)

m = classification(X,Y.flatten(),k)
m.randomize()#
m.optimize('bfgs')
m.plot()
