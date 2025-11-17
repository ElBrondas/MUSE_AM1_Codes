from numpy import zeros


def Cauchy_problem(F, U0, t, Temporal_scheme):

    N = len(t) - 1 # t0, t1, ..., tN is of length N+1
    Nv = len(U0) # Number of variables

    U = zeros((N+1, Nv))
    U[0,:] = U0

    for n in range(N):
        U[n+1,:] = Temporal_scheme(U[n,:], t[n], t[n+1], F)
    return U