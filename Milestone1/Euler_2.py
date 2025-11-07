"""
Resolve two-body problem using Euler method and plot the trajectory.
"""

from numpy import array, concatenate, zeros
from numpy.linalg import norm
import matplotlib.pyplot as plt

def F(U, t1): # F:R4 -> R4
    """
    Returns the derivative of the state vector U.
    """
    r = U[0:2]
    rdot = U[2:4]
    return concatenate((rdot, -r/norm(r)**3), axis=None) # In this case, we use None because there is no ambiguity

def Euler(U, t1, t2, F):
    return U + (t2 - t1) * F(U, t1)

N = 10000
dt = 0.01

U = zeros((N+1, 4)) 

U0 = array([1.0, 0.0, 0.0, 1.0]) 
U[0,:] = U0
for n in range(0, N):
    tn = n * dt
    tn1 = (n + 1) * dt
    U[n+1,:] = Euler(U[n,:], tn, tn1, F)

# for n in range(0, N):         
#     U[n+1,:] = U[n,:] + dt * F(U[n,:])
    
# Plot trajectory
plt.plot(U[:,0], U[:,1])  # Column 0 = x, Column 1 = y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectory in the XY plane')
plt.axis('equal') 
plt.grid(True)
plt.show()

