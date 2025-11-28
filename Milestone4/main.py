"""
Poner API
"""

from numpy import linspace, concatenate, array, meshgrid, transpose
from numpy.linalg import norm
from Cauchy import Cauchy_problem, Cauchy_problem_2_steps
from Temporal_schemes import *
from Convergence_and_Stability import convergence_rate, stability_region
from Physics import *
import matplotlib.pyplot as plt


def test_Cauchy_1_step():

    # Variables
    T = 20
    N = 10000
    t = linspace(0, T, N+1)

    # Initial conditions
    U0 = ([0, 1])        

    U = Cauchy_problem(Oscilador, U0, t, LeapFrog(U0, Oscilador))
    
    plt.plot(U[:, 0], U[:, 1]) 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory in the XY plane')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def test_Cauchy_2_steps():

    # Variables
    T = 20
    N = 10000
    t = linspace(0, T, N+1)

    # Initial conditions
    U0 = ([0, 1]) 
    U1 = Euler(U0, t[0], t[1], Oscilador)           

    U = Cauchy_problem_2_steps(Oscilador, U0, U1, t, LeapFrog_v2)
    
    plt.plot(U[:, 0], U[:, 1])  
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory in the XY plane')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def test_Stability():
    
    rho, x, y = stability_region(CrankNicolson)    
    
    plt.contour(x, y, transpose(rho), linspace(0, 1, 11)) # Pinto entre rho = [0,1]
    plt.xlabel('Re(w)')
    plt.ylabel('Im(w)')
    plt.axis('equal')
    plt.grid()
    plt.show()


# test_Cauchy_1_step()
# test_Cauchy_2_steps()
test_Stability()