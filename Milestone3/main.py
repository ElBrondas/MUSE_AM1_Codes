from numpy import linspace, concatenate
from numpy.linalg import norm
from Cauchy import Cauchy_problem
from Error import Cauchy_error
from Temporal_schemes import *
import matplotlib.pyplot as plt


def F(U, t):

    r = U[0:2]
    rdot = U[2:4]
    return concatenate((rdot, -r/norm(r)**3), axis=None)


def test_Error():

    # Variables
    U0 = ([1, 0, 0, 1])
    T = 20
    N = 10000
    t = linspace(0, T, N+1)
    q = 1

    U, E = Cauchy_error(F, U0, t, CrankNicolson, q)

    # Plot trajectory
    plt.plot(U[:, 0], U[:, 1])  # Column 0 = x, Column 1 = y
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory in the XY plane')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    plt.plot(t, E[:, 0])  # Column 0 = x, Column 1 = y
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.title('Error in the XY plane')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


test_Error()