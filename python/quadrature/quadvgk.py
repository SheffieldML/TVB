import numpy as np
import pylab as pb
def quadvgk(f, Subs, NF):
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
    #G = (2:2:NK)';  % 7-point Gaussian poisitions (Subset of the Kronrod points)
    G = np.arange(2,NK,2)

    def GetSubs(Subs):
        M = (Subs[1,:]-Subs[0,:])/2.
        C = (Subs[1,:]+Subs[0,:])/2.
        I = XK[:,None]*M + np.ones((NK,1))*C
        A = np.vstack((Subs[0,:], I))
        B = np.vstack((I, Subs[1,:]))
        Subs = np.vstack((A.flatten(), B.flatten()))
        return Subs

    Q = np.zeros(NF)
    while Subs.size:
        Subs = GetSubs(Subs)
        M = (Subs[1,:]-Subs[0,:])/2.
        C = (Subs[1,:]+Subs[0,:])/2.
        NM = M.size
        x = (XK[:,None]*M + C).flatten()

        FV = f(x);
        Q1 = np.zeros((NF, NM));
        Q2 = np.zeros((NF, NM));
        for n in range(NF):
            F = FV[n].reshape(NK, -1)
            Q1[n,:] = M*np.dot(F.T, WK)
            Q2[n,:] = M*np.dot(F[G,:].T, WG)
        stop
        ind = np.nonzero(np.logical_or(np.max(np.abs((Q1-Q2)/Q1), 0) < 1e-6 , (Subs[1,:] - Subs[0,:]) < 1e-16))[0]
        print Subs.shape
        Q = Q + np.sum(Q1[:,ind], 1);
        Subs = np.delete(Subs, ind, axis=1)
    return Q

if __name__=='__main__':
    def f(x):
        n = np.array([1., 2., 3.])
        return 1./(n[:,None]+x[None,:])
    from scipy.integrate import quad
    print quad(lambda x : 1./(1.+x), 1,2)
    print quad(lambda x : 1./(2.+x), 1,2)
    print quad(lambda x : 1./(3.+x), 1,2)

    print quadvgk(f, np.array([[1.],[2.]]),3)

