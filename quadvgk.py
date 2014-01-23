# Copyright (c) 2014, James Hensman
# Distributed under the terms of the GNU General public License, see LICENSE.txt

import numpy as np
import pylab as pb
def GetSubs(Subs, XK):
    M = (Subs[1,:]-Subs[0,:])/2.
    C = (Subs[1,:]+Subs[0,:])/2.
    I = XK[:,None]*M + C
    A = np.vstack((Subs[0,:], I))
    B = np.vstack((I, Subs[1,:]))
    Subs = np.vstack((A.flatten(), B.flatten()))
    return Subs

def quadvgk(f, a, b, difftol=1e-4, xtol=1e-4):
    """
    integrate the functions(s) f from a to b, where a and ba are finite.

    The function f may include several (prefereably related functions). f takes
    a 1D vector (length NX) and returns a NF x NM 2D array, containing the
    value of each function at each requested point. 

    for infinite integration, see inf_quadvgk

    This code uses the gauss-kronrod method. Adapted from:
    http://www.mathworks.co.uk/matlabcentral/fileexchange/18801-quadvgk
    """
    XK = np.array([-0.991455371120813, -0.949107912342759, -0.864864423359769, -0.741531185599394,
                   -0.586087235467691, -0.405845151377397, -0.207784955007898, 0.,
                   0.207784955007898, 0.405845151377397, 0.586087235467691,
                   0.741531185599394, 0.864864423359769, 0.949107912342759, 0.991455371120813])
    WK = np.array([0.022935322010529, 0.063092092629979, 0.104790010322250, 0.140653259715525,
                   0.169004726639267, 0.190350578064785, 0.204432940075298, 0.209482141084728,
                   0.204432940075298, 0.190350578064785, 0.169004726639267,
                   0.140653259715525, 0.104790010322250, 0.063092092629979, 0.022935322010529])
     # 7-point Gaussian weightings
    WG = np.array([0.129484966168870, 0.279705391489277, 0.381830050505119, 0.417959183673469,
        0.381830050505119, 0.279705391489277, 0.129484966168870])
    NK = WK.size
    #index_G = (2:2:NK)';  % 7-point Gaussian poisitions (Subset of the Kronrod points)

    Subs = np.array([[a],[b]])

    NF = f(np.zeros(1)).size
    Q = np.zeros(NF)
    neval = 0
    while Subs.size:
        Subs = GetSubs(Subs, XK)
        M = (Subs[1,:]-Subs[0,:])/2.
        C = (Subs[1,:]+Subs[0,:])/2.
        NM = M.size
        x = (XK[:,None]*M + C).flatten()

        FV = f(x)
        neval += x.size
        Q1 = np.dot(FV.reshape(NF, NK, NM).swapaxes(2,1),WK)*M
        Q2 = np.dot(FV.reshape(NF, NK, NM).swapaxes(2,1)[:,:,1::2],WG)*M
        #ind = np.nonzero(np.logical_or(np.max(np.abs((Q1-Q2)/Q1), 0) < difftol , M < xtol))[0]
        ind = np.nonzero(np.logical_or(np.max(np.abs((Q1-Q2)), 0) < difftol , M < xtol))[0]
        Q = Q + np.sum(Q1[:,ind], 1)
        Subs = np.delete(Subs, ind, axis=1)
    return Q, neval

def inf_quadvgk(f, difftol=1e-3, xtol=5e-3):
    """
    integrate f(x) from -inf to inf, by changing the variable:
    x = r/(1-r^2)
    so the limits become r = -1, 1

    See also:
      quadvgk
    """
    def g(r):
        r2 = np.square(r)
        x = r/(1.-r2)
        return f(x)*(r2+1.)/np.square(r2-1.)
    return quadvgk(g, -1, 1, difftol, xtol)

if __name__=='__main__':
    N = 10
    def f(x):
        n = np.arange(1, N+1)
        return 1./(n[:,None]+x[None,:])

    print quadvgk(f, 1., 200.)

    from scipy.integrate import quad
    for n in range(N):
        print quad(lambda x : 1./(n+1.+x), 1,200)

    #do an infinte integral thing:
    def f(x):
        return np.exp(-0.5*x**2)*np.power(x,np.arange(5)[:,None])
    print inf_quadvgk(f)


