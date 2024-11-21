import numpy as np
import matplotlib.pyplot as plt
import MatrixOperations as mat

#Cached Rho Equation
#Operates on d_funcs and lambda_funcs operating exclusively on the spacial variable x

"""
Cached Thomas Class

Runs the thomas algorithm but caches the left hand side, along with scale factor

initial_pass: Run the first time in order to required vectors prior to subsequent pass

subsequent_pass: Runs the thomas algorithm on the vectors
"""
class CachedThomas():
    def __init__(self):
        pass
    
    def initial_pass(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.len = len(a)
        self.scale_factor = np.zeros(self.len - 1, dtype="float64")

        for i in range(1, self.len):
            if abs(self.b[i - 1]) < 1e-12:
                raise ValueError("Numerical instability: Main diagonal element is too small.")
            self.scale_factor[i - 1] = a[i] / b[i - 1]  # Scale factor
            self.b[i] = b[i] - self.scale_factor[i - 1] * c[i - 1]  # Modify the main diagonal
            
        return self.subsequent_pass(d)
          
    def subsequent_pass(self, d):
        for i in range(1, self.len):
            d[i] = d[i] - self.scale_factor[i - 1] * d[i - 1]  # Modify the result vector

        x = np.zeros(self.len, dtype="float64")
            # Back substitution
        x[-1] = d[-1] / self.b[-1]  # Solve last unknown 

        for i in range(self.len - 2, -1, -1):
            x[i] = (d[i] - self.c[i] * x[i + 1]) / self.b[i]  # Solve for the other unknowns
    
        return x

"""
Rho Equation Solver

Sol
""" 
def solve_rho_eq(start_space = 0, end_space=10.0, start_time = 0, end_time = 10.0, dx = .001, dt = .1, lambda_func=lambda x: 0.5, D_func=lambda x: 1, rho_func=lambda X: np.sin(np.pi * X) + 1):
    num_x = round((end_space - start_space) / dx)
    num_t = round((end_time - start_time) / dt)

    # Spatial and time grids
    X = np.linspace(start_space, end_space, num_x, dtype="float64")  # Goes from 0 to L, generating num_x samples
    # Initialize rho (initial condition)
    rho = np.zeros((num_t, num_x))
    rho[0 , :] = rho_func(X)  # Initial condition, e.g., a wave

    # Coefficients for the tridiagonal matrix A, sub, main, super
    a = np.zeros(num_x, dtype="float64")
    b = np.zeros(num_x, dtype="float64")
    c = np.zeros(num_x, dtype="float64") 
    d = np.zeros(num_x, dtype="float64")

    thomas_algo = CachedThomas()
    # Time stepping loop
    
        #separate calculation, i = 0
    lambda_i = lambda_func(X[0])
    D_i_plus_half = (D_func(X[0]) + D_func(X[1])) / 2.0
    b[0] = 1 + (dt / (dx**2)) * (D_i_plus_half)
    c[0] = -dt * D_i_plus_half / dx**2
    d[0] = rho[0, 0] * (1 - dt * lambda_i)
        
    # Internal points: 1 to num_x - 2
    for i in range(1, num_x-1):
        D_i_plus_half = (D_func(X[i]) + D_func(X[i + 1])) / 2.0
        D_i_minus_half = (D_func(X[i]) + D_func(X[i - 1])) / 2.0
        lambda_i = lambda_func(X[i])

        # Populate tridiagonal matrix elements for internal points
        a[i] = -dt * D_i_minus_half / dx**2  # p_i-1 coefficient
        b[i] = 1 + (dt / (dx**2)) * (D_i_plus_half + D_i_minus_half)  # p_i coefficient
        c[i] = -dt * D_i_plus_half / dx**2  # p_i+1 coefficient

        # Fill the right-hand side d vector
        d[i] = rho[0, i] * (1 - dt * lambda_i)
        
    #Separate calculation, i = num_x - 1
    lambda_i = lambda_func(X[num_x-1])
    D_i_minus_half = (D_func(X[num_x - 1]) + D_func(X[num_x - 2])) / 2.0
    a[-1] = -dt * D_i_minus_half / dx**2 
    b[-1] = 1 + (dt / (dx**2)) * (D_i_minus_half)
    d[-1] = rho[0, -1] * (1 - dt * lambda_i)

    rho[1] = thomas_algo.initial_pass(a, b, c, d)

        
        #subsequent cached calculations
    for n in range(1, num_t - 1):
        d[0] = rho[n, 0] * (1 - dt * lambda_i)
        for i in range(1, num_x-1):
            d[i] = rho[n, i] * (1 - dt * lambda_i)
        d[-1] = rho[n, -1] * (1 - dt * lambda_i)
        rho[n + 1] = thomas_algo.subsequent_pass(d)
        
    return X, rho
    
def main():
    test_sin_neumann()
    
def test_sin_neumann():
    start_space = 0
    end_space=10.0
    start_time = 0
    end_time = 10.0
    dx = .001
    dt = .1
    
    num_x = round((end_space - start_space) / dx)
    num_t = round((end_time - start_time) / dt)

    X, rho = solve_rho_eq(start_space = start_space, end_space = end_space, start_time = start_time, end_time = end_time, dx = dx, dt = dt, lambda_func=lambda x: 0, D_func=lambda x: .5, rho_func=lambda X: np.sin(np.pi * X) + 1)
    
    for i in range(0, num_t, int(num_t // (end_time - start_time))):
        plt.plot(X, rho[i, :], label=f"t={i * (end_time - start_time) / num_t:.3f}")

    plt.xlabel('Position (x)')
    plt.ylabel('Density (rho)')
    plt.legend()
    plt.title('1D Diffusion with Variable Coefficients and Neumann Boundary Conditions Included in Matrix')
    plt.show()
 
if __name__ == "__main__":
    main()