"""
Resolve two-body problem using Crank-Nicolson method and plot the trajectory.
"""

from numpy import array, concatenate, zeros, zeros_like, eye
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt

def F(U): # F:R4 -> R4
    """
    Returns the derivative of the state vector U.

    :param U: State vector.
    :type U: numpy.ndarray, shape (4,)

    :return: Derivative of the state vector.
    :rtype: numpy.ndarray, shape (4,)
    """
    r = U[0:2]
    rdot = U[2:4]
    return concatenate((rdot, -r/norm(r)**3), axis=None) # In this case, we use None because there is no ambiguity

def Jacobian_F_Analytic(U): # J:R4 -> R4x4
    """
    Returns the analytic Jacobian matrix of F at U.

    :param U: State vector.
    :type U: numpy.ndarray, shape (4,)
    
    :return: Jacobian matrix.
    :rtype: numpy.ndarray, shape (4, 4)
    """
    r = U[0:2]
    x = r[0]
    y = r[1]    
    
    J = zeros((4, 4))
    J[0, 2] = 1
    J[1, 3] = 1
    J[2, 0] = (3 * x**2 - norm(r)**2) / norm(r) ** 5
    J[2, 1] = 3 * x * y / norm(r) ** 5
    J[3, 0] = 3 * x * y / norm(r) ** 5
    J[3, 1] = (3 * y**2 - norm(r)**2) / norm(r) ** 5

    return J

def Jacobian_F_Numeric(U, dt): # J:R4 -> R4x4
    """
    Returns the numeric Jacobian matrix of F at U using finite differences.

    :param U: State vector.
    :type U: numpy.ndarray, shape (4,)
    
    :return: Jacobian matrix.
    :rtype: numpy.ndarray, shape (4, 4)
    """        
    J = zeros((4, 4))
    for j in range(4):   
        dU = zeros_like(U)  
        dU[j] = dt   
        J[:, j] = (F(U + dU) - F(U - dU)) / (2 * dt) # Central difference

    return J

def R(U_next, U, dt): # R:R4 -> R4
    """
    Computes the residual for the Crank-Nicolson method.

    :param U_next: Guess for the next state vector.
    :type U_next: numpy.ndarray, shape (4,)
    :param U: Known previous state vector.
    :type U: numpy.ndarray, shape (4,)
    
    :return: Residual vector.
    :rtype: numpy.ndarray, shape (4,)
    """    
    return U_next - U - dt / 2 * (F(U_next) + F(U))

def Jacobian_R_Analytic(U, dt): # J:R4 -> R4x4
    """
    Computes the analytic Jacobian of the residual for the Crank-Nicolson method.
    
    :param U: Known previous state vector.
    :type U: numpy.ndarray, shape (4,)
    
    :return: Jacobian matrix.
    :rtype: numpy.ndarray, shape (4, 4)
    """    
    return eye(4) - dt / 2 * Jacobian_F_Analytic(U)

def Jacobian_R_Numeric(U, dt): # J:R4 -> R4x4
    """
    Computes the numeric Jacobian of the residual for the Crank-Nicolson method.
    
    :param U: Known previous state vector.
    :type U: numpy.ndarray, shape (4,)
    
    :return: Jacobian matrix.
    :rtype: numpy.ndarray, shape (4, 4)
    """    
    return eye(4) - dt / 2 * Jacobian_F_Numeric(U, dt)

# Variables
U0 = ([1, 0, 0, 1])
T = 200
N = 10000
dt = T/N
eps = 1e-10

U = zeros((N+1, 4)) 
U[0,:] = U0

for n in range(0, N):  
    # Define candidate 
    U_guess = U[n, :] + dt * F(U[n, :]) # Explicit Euler step as initial guess

    for _ in range(100): # Newton-Raphson iterations
        JR = Jacobian_R_Analytic(U_guess, dt) 
        delta = solve(JR, -R(U_guess, U[n, :], dt))
        U_guess = U_guess + delta
        if norm(delta) < eps:
            break
    
    U[n+1,:] = U_guess


# Plot trajectory
plt.plot(U[:, 0], U[:, 1])  # Column 0 = x, Column 1 = y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectory in the XY plane')
plt.axis('equal') 
plt.grid(True)
plt.show()


