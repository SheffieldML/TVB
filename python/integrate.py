import numpy as np
from quadvgk import inf_quadvgk

SQRT_2PI = np.sqrt(2.*np.pi)
LOG_SQRT_2PI = np.log(SQRT_2PI)

def integrate(Y, m, s, lik, derivs=True):
    """
    compute multiple integrals related to the tilted distribution

    This function generates and then integrates a function f:

    f accepts a vector of inputs (1D numpy array), representing points of
    the variable of integration.  f returns a matrix representing several
    functions evaluated at those points. Let c(x) be a the 'cavity'
    distribution, c(x) = N(x|m, s**2), p(y|x) is a likelihood, then f
    computes the following functions on x
    p(y|x) * c(x)
    p(y|x) * c(x) * x
    p(y|x) * c(x) * x**2
    p(y|x) * c(x) * x**3
    p(y|x) * c(x) * x**4

    if derivs is true, we also stack in:

    p(y|x) * c(x) * d(ln p(y|x) / d theta)
    p(y|x) * c(x) * d(ln p(y|x) / d theta) + x
    p(y|x) * c(x) * d(ln p(y|x) / d theta) + x**2

    where theta is some parameter of the likelihood (e.g. the std of the noise, or the degrees of freedom)

    If there are several parameters which require derivatives, then we have
    multiple lines for each.

    Once we've genreated the function, we integrate using quadvgk.
    This function appears separately for ease of parallelisation.

    """
    assert np.array(Y).size==1, "we're only doing 1 data point at a time"
    if derivs:
        def f(x):
            a = lik.pdf(x, Y) * np.exp(-0.5*np.square((x-m)/s))/SQRT_2PI/s
            p = np.power(x, np.arange(5)[:,None])
            pp = np.tile(p[:3], [2, 1]) # TODO! make the number of params configurble (currently 2)
            derivs = lik.dlnpdf_dtheta(x, Y).repeat(3,0)
            return a * np.vstack((p, pp*derivs))[:,None]
    else:
        def f(x):
            return lik.pdf(x, Y) * np.exp(-0.5*np.square((x-m)/s))/SQRT_2PI/s * np.power(x, np.arange(5)[:,None])
    return inf_quadvgk(f)


