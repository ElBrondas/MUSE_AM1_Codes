"""
Resolve two-body problem using Euler method and plot the trajectory.
"""

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

N = 10
dt = 0.001

U = zeros((N, 4))

U0 = array([1.0, 0.0, 0.0, 1.0]) 
U[0] = U0

for i in range(1, N):
    U[i] = U[i-1] + dt * F(U[i-1])
  
# Plot trajectory
plt.plot(U[:,0], U[:,1])  # Column 0 = x, Column 1 = y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectory in the XY plane')
plt.axis('equal') 
plt.grid(True)
plt.show()

