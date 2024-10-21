import numpy as np
import matplotlib.pyplot as plt
import MatrixOperations as mat

"""
Returns X values and Density values using Finite Difference combined with thomas algorithm
L: Total Space
T: Total Time
num_x: Number of spatial points
num_t: Number of time steps
lambda_func: Decay function
D_func: Diffusion function

"""

def solve_rho_eq(L = 1.0 , T = 1.0, num_x = 50, num_t = 100, lambda_func =  lambda x, t: 0.1, D_func = lambda x, t: 1, rho_func = lambda X : np.sin(np.pi * X)):
    #Deltas
    dx = L / (num_x - 1)  # Spatial step size. -1 because of boundary conditions
    dt = T / num_t  # Time step size

    # Spatial and time grids
    # Linspace equally distributes
    X = np.linspace(0, L, num_x) #Goes from 0 to L, generationg num_x samples
    T = np.linspace(0, T, num_t) # Goes from 0 to T, generating num_t samples

    # Initialize rho (initial condition)
    rho = np.zeros((num_t, num_x))
    rho[0, :] = rho_func(X) # Initial condition, e.g., a wave. rho == density

    # Time stepping loop

    for n in range(0, num_t-1):
        # Coefficients for the tridiagonal matrix A
        a = np.zeros(num_x-2, dtype="float64")  # Subdiagonal (for i-1). a[0] == 0
        b = np.zeros(num_x-2, dtype="float64")  # Main diagonal (for i)
        c = np.zeros(num_x-2, dtype="float64")  # Superdiagonal (for i+1) c[-1] == 0
        d = np.zeros(num_x-2, dtype="float64")  # Right-hand side (known values from previous time step)
        
        # ignore boundaries: 0 and num_x -1
        for i in range(1, num_x-1):
            D_i_plus_half = (D_func(X[i], T[n]) + D_func(X[i+1], T[n])) / 2
            D_i_minus_half = (D_func(X[i], T[n]) + D_func(X[i-1], T[n])) / 2
            lambda_i = lambda_func(X[i], T[n])
            
            if 1 <= i < num_x-1:
                #Get Tridiagonal matrix Ax = d
                
                #subdiagonal (a), main diagonal (b), and superdiagonal (c)
                a[i-1] = -dt * D_i_minus_half / dx**2 #p_i-1 coefficient
                b[i-1] = 1 + dt * (2 * D_func(X[i], T[n]) / dx**2 + lambda_i) #p_i Coefficient
                c[i-1] = -dt * D_i_plus_half / dx**2  #p_i+1
                
                # Fill the right-hand side d vector
                d[i-1] = rho[n, i]
                
        # Solve the tridiagonal system for the next time step
        rho[n+1, 1:-1] = mat.solve_matrix_thomas(a, b, c, d)
        
    return X, rho

def main():
    num_t = 100
    T = 1.0

    X, rho = solve_rho_eq(num_t=num_t, T=T)

    for i in range(0, num_t, num_t//10):
        plt.plot(X, rho[i, :], label=f"t={i*T / num_t:.3f}")   
        
    plt.xlabel('Position (x)')
    plt.ylabel('Density (rho)')
    plt.legend()
    plt.title('1D Diffusion with Variable Coefficients')
    plt.show()
    
if __name__ == "main":
    main() 
