from numpy import concatenate, array, reshape, zeros
from numpy.linalg import norm

def F(U, t):

    r = U[0:2]
    rdot = U[2:4]

    return concatenate((rdot, -r/norm(r)**3), axis=None)


def Oscilador(U, t):
    """
    d2x/dt2 + x = 0
    """

    x = U[0]
    xdot = U[1]

    return array((xdot, -x))