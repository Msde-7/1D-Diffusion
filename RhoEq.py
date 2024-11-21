import numpy as np
import matplotlib.pyplot as plt
import MatrixOperations as mat

def solve_rho_eq(start_space = 0, end_space=10.0, start_time = 0, end_time = 10.0, dx = .001, dt = .1, lambda_func=lambda x, t: 0.5, D_func=lambda x, t: 1, rho_func=lambda X: np.sin(np.pi * X) + 1):
    num_x = round((end_space - start_space) / dx)
    num_t = round((end_time - start_time) / dt)

    # Spatial and time grids
    X = np.linspace(start_space, end_space, num_x, dtype="float64")  # Goes from 0 to L, generating num_x samples
    T = np.linspace(start_time, end_time, num_t, dtype="float64")  # Goes from 0 to T, generating num_t samples

    # Initialize rho (initial condition)
    rho = np.zeros((num_t, num_x))
    rho[0 , :] = rho_func(X)  # Initial condition, e.g., a wave

    # Coefficients for the tridiagonal matrix A, sub, main, super
    a = np.zeros(num_x, dtype="float64")
    b = np.zeros(num_x, dtype="float64")
    c = np.zeros(num_x, dtype="float64") 
    d = np.zeros(num_x, dtype="float64")
    
    # Time stepping loop
    for n in range(0, num_t - 1):
        #separate calculation, i = 0
        lambda_i = lambda_func(X[0], T[n])
        D_i_plus_half = (D_func(X[0], T[n]) + D_func(X[1], T[n])) / 2.0
        b[0] = 1 + (dt / (dx**2)) * (D_i_plus_half)
        c[0] = -dt * D_i_plus_half / dx**2
        d[0] = rho[n, 0] * (1 - dt * lambda_i)
        
        
        # Internal points: 1 to num_x - 2
        for i in range(1, num_x-1):
            D_i_plus_half = (D_func(X[i], T[n]) + D_func(X[i + 1], T[n])) / 2.0
            D_i_minus_half = (D_func(X[i], T[n]) + D_func(X[i - 1], T[n])) / 2.0
            lambda_i = lambda_func(X[i], T[n])

            # Populate tridiagonal matrix elements for internal points
            a[i] = -dt * D_i_minus_half / dx**2  # p_i-1 coefficient
            b[i] = 1 + (dt / (dx**2)) * (D_i_plus_half + D_i_minus_half)  # p_i coefficient
            c[i] = -dt * D_i_plus_half / dx**2  # p_i+1 coefficient

            # Fill the right-hand side d vector
            d[i] = rho[n, i] * (1 - dt * lambda_i)
        
        #Separate calculation, i = num_x - 1
        lambda_i = lambda_func(X[num_x-1], T[n])
        D_i_minus_half = (D_func(X[num_x - 1], T[n]) + D_func(X[num_x - 2], T[n])) / 2.0
        a[-1] = -dt * D_i_minus_half / dx**2 
        b[-1] = 1 + (dt / (dx**2)) * (D_i_minus_half)
        d[-1] = rho[n, -1] * (1 - dt * lambda_i)
        
        rho[n + 1] = mat.solve_matrix_thomas(a, b, c, d)

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

    X, rho = solve_rho_eq(start_space = start_space, end_space = end_space, start_time = start_time, end_time = end_time, dx = dx, dt = dt, lambda_func=lambda x, t: 0, D_func=lambda x, t: .5, rho_func=lambda X: np.sin(np.pi * X) + 1)
    
    for i in range(0, num_t, int(num_t // (end_time - start_time))):
        plt.plot(X, rho[i, :], label=f"t={i * (end_time - start_time) / num_t:.3f}")

    plt.xlabel('Position (x)')
    plt.ylabel('Density (rho)')
    plt.legend()
    plt.title('1D Diffusion with Variable Coefficients and Neumann Boundary Conditions Included in Matrix')
    plt.show()
    

if __name__ == "__main__":
    main()
