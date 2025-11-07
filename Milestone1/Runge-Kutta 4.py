from numpy import array, concatenate, zeros
from numpy.linalg import norm
import matplotlib.pyplot as plt

def F(U): # F:R4 -> R4
    """
    Returns the derivative of the state vector U.
    """
    r = U[0:2]
    rdot = U[2:4]
    return concatenate((rdot, -r/norm(r)**3), axis=None) # In this case, we use None because there is no ambiguity

def RungeKutta4(U, dt, F):
    """
    Performs a single step of the 4th-order Runge-Kutta method.

    :param U: Current state vector.
    :type U: numpy.ndarray, shape (4,)
    :param dt: Time step.
    :type dt: float
    :param F: Function that computes the derivative of U.
    :type F: callable

    :return: Updated state vector after one time step.
    :rtype: numpy.ndarray, shape (4,)
    """
    k1 = F(U)
    k2 = F(U + dt/2 * k1)   
    k3 = F(U + dt/2 * k2)  
    k4 = F(U + dt * k3)

    return U + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Variables
U0 = ([1, 0, 0, 1])
T = 2000
N = 10000
dt = T/N

U = zeros((N+1, 4)) 
U[0,:] = U0

# for n in range(0, N):    
#     k1 = F(U[n,:])
#     k2 = F(U[n,:] + dt/2 * k1)   
#     k3 = F(U[n,:] + dt/2 * k2)  
#     k4 = F(U[n,:] + dt * k3)

#     U[n+1,:] = U[n,:] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Plot trajectory
plt.plot(U[:, 0], U[:, 1])  # Column 0 = x, Column 1 = y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectory in the XY plane')
plt.axis('equal') 
plt.grid(True)
plt.show()