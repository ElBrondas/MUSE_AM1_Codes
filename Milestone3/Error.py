from numpy import zeros, linspace
from Cauchy import Cauchy_problem


def Cauchy_error(F, U0, t, Temporal_scheme, q):

    N = len(t) - 1
    Nv = len(U0)

    E = zeros((N+1, Nv))
    t1 = t[:]
    t2 = linspace(t[0], t[N], 2*N+1)

    U1 = Cauchy_problem(F, U0, t1, Temporal_scheme)
    U2 = Cauchy_problem(F, U0, t2, Temporal_scheme)

    for n in range(0, N+1):
        E[n,:] = (U2[2*n,:] - U1[n,:])/(1-1/2**q)

    return U1, E