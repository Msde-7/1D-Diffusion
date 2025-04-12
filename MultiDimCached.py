import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import MatrixOperations as mat
import matplotlib.animation as animation



"""
-Cached Thomas Class-
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
-Cached Thomas 2D Class-
Stores individual cached thomas algorithms for each x and y, required to cache in 2d
"""
class CachedThomas2D():
    def __init__(self, num_x, num_y):
        self.x_cache = []
        self.y_cache = []
        for i in range(num_x):
            self.x_cache.append(CachedThomas())
        for i in range(num_y):
            self.y_cache.append(CachedThomas())

    def initial_pass_x(self, a, b, c, d, x_cache_idx):
        return self.x_cache[x_cache_idx].initial_pass(a, b, c, d)
    
    def initial_pass_y(self, a, b, c, d, y_cache_idx):
        return self.y_cache[y_cache_idx].initial_pass(a, b, c, d)
    
    def subsequent_pass_x(self, d, x_cache_idx):
        return self.x_cache[x_cache_idx].subsequent_pass(d)
    
    def subsequent_pass_y(self, d, y_cache_idx):
        return self.y_cache[y_cache_idx].subsequent_pass(d)

"""
Rho Equation Solver

Solves the rho equation in 2D using a cached Thomas algorithm
Only possible with constant D and lambda
""" 
def solve_rho_eq_2d_cached(start_space_x=0, end_space_x=10.0,
                      start_space_y=0, end_space_y=10.0,
                      start_time=0, end_time=10.0,
                      dx=0.1, dy=0.1, dt=0.1,
                      lambda_func=lambda x, y, t: 0,
                      D_func=lambda x, y, t: 1,
                      rho_func=lambda X, Y: (np.sin(np.pi * X) * np.sin(np.pi * Y))):
    # Compute number of grid points
    num_x = round((end_space_x - start_space_x) / dx)
    num_y = round((end_space_y - start_space_y) / dy)
    num_t = round((end_time - start_time) / dt)
    
    # Create spatial and time grids
    X = np.linspace(start_space_x, end_space_x, num_x, dtype="float64")
    Y = np.linspace(start_space_y, end_space_y, num_y, dtype="float64")
    T = np.linspace(start_time, end_time, num_t, dtype="float64")
    
    # Solutions rho[time][x][y]
    rho = np.zeros((num_t, num_x, num_y), dtype="float64")
    # Set the initial condition using a meshgrid
    X_grid, Y_grid = np.meshgrid(X, Y, indexing='ij')
    rho[0, :, :] = rho_func(X_grid, Y_grid)
    #--Exact same up to here

    # Initialize CachedThomas2D with correct dimensions
    cachedThomas2D = CachedThomas2D(num_x, num_y)
    
    # Temporary buffer
    rho_temp = np.zeros((num_x, num_y), dtype="float64")

    # First solve X and Y sweeps at t=0, caching results

    # X-Sweep for t = 0
    for j in range(num_y):
        a = np.zeros(num_x, dtype="float64")
        b = np.zeros(num_x, dtype="float64")
        c = np.zeros(num_x, dtype="float64")
        d = np.zeros(num_x, dtype="float64")

        # Left boundary
        D_ip = (D_func(X[0], Y[j]) + D_func(X[1], Y[j])) / 2.0
        b[0] = 1 + (dt / dx**2) * D_ip
        c[0] = - dt * D_ip / dx**2
        d[0] = rho[0, 0, j] * (1 - dt * lambda_func(X[0], Y[j]))

        # Internal points
        for i in range(1, num_x - 1):
            D_ip = (D_func(X[i], Y[j]) + D_func(X[i+1], Y[j])) / 2.0 
            D_im = (D_func(X[i], Y[j]) + D_func(X[i-1], Y[j])) / 2.0
            a[i] = - dt * D_im / dx**2
            b[i] = 1 + dt / dx**2 * (D_ip + D_im)
            c[i] = - dt * D_ip / dx**2
            d[i] = rho[0, i, j] * (1 - dt * lambda_func(X[i], Y[j]))

        # Right boundary
        D_im = (D_func(X[num_x-1], Y[j]) + D_func(X[num_x-2], Y[j])) / 2.0
        a[-1] = - dt * D_im / dx**2 
        b[-1] = 1 + dt * D_im / dx**2
        d[-1] = rho[0, num_x-1, j] * (1 - dt * lambda_func(X[num_x-1], Y[j]))

        rho_temp[:, j] = cachedThomas2D.initial_pass_y(a, b, c, d, y_cache_idx=j)

    # Y-Sweep for t = 0
    rho_new = np.zeros((num_x, num_y), dtype="float64")
    for i in range(num_x):
        a = np.zeros(num_y, dtype="float64")
        b = np.zeros(num_y, dtype="float64")
        c = np.zeros(num_y, dtype="float64")
        d = np.zeros(num_y, dtype="float64")

        # Bottom boundary
        D_jp = (D_func(X[i], Y[0]) + D_func(X[i], Y[1])) / 2.0
        b[0] = 1 + (dt / dy**2) * D_jp
        c[0] = - dt * D_jp / dy**2
        d[0] = rho_temp[i, 0] * (1 - dt * lambda_func(X[i], Y[0]))

        # Internal points
        for j in range(1, num_y - 1):
            D_jp = (D_func(X[i], Y[j]) + D_func(X[i], Y[j+1])) / 2.0
            D_jm = (D_func(X[i], Y[j]) + D_func(X[i], Y[j-1])) / 2.0
            a[j] = - dt * D_jm / dy**2
            b[j] = 1 + dt / dy**2 * (D_jp + D_jm)
            c[j] = - dt * D_jp / dy**2
            d[j] = rho_temp[i, j] * (1 - dt * lambda_func(X[i], Y[j]))

        # Top boundary
        D_jm = (D_func(X[i], Y[num_y-1]) + D_func(X[i], Y[num_y-2])) / 2.0
        a[-1] = - dt * D_jm / dy**2
        b[-1] = 1 + dt * D_jm / dy**2
        d[-1] = rho_temp[i, num_y-1] * (1 - dt * lambda_func(X[i], Y[num_y-1]))

        rho_new[i, :] = cachedThomas2D.initial_pass_x(a, b, c, d, x_cache_idx=i)

    # Store the result of t = 1
    rho[1] = rho_new

    # Calculate subsequent time steps using cached results
    for n in range(1, num_t-1):
        rho_temp = np.zeros((num_x, num_y), dtype="float64")
        rho_new = np.zeros((num_x, num_y), dtype="float64")
        # Y-sweep
        for j in range(num_y):
            d = np.zeros(num_x, dtype="float64")  # Reset d for each row
            for i in range(num_x):
                d[i] = rho[n, i, j] * (1 - dt * lambda_func(X[i], Y[j]))
            rho_temp[:, j] = cachedThomas2D.subsequent_pass_y(d, y_cache_idx=j)

        # X-sweep
        for i in range(num_x):
            d = np.zeros(num_y, dtype="float64")  # Reset d for each column
            for j in range(num_y):
                d[j] = rho_temp[i, j] * (1 - dt * lambda_func(X[i], Y[j]))
            rho_new[i, :] = cachedThomas2D.subsequent_pass_x(d, x_cache_idx=i)

        rho[n+1] = rho_new.copy()

    return X, Y, rho

    


def test_diffusion_2d():
    start_space_x = 0
    end_space_x = 10.0
    start_space_y = 0
    end_space_y = 11.0
    start_time = 0
    end_time = 10.0

    #orig .1 all
    dx = 0.1
    dy = 0.1
    dt = 0.1
    
    """
    X, Y, rho = solve_rho_eq_2d_cached(start_space_x, end_space_x,
                                  start_space_y, end_space_y,
                                  start_time, end_time,
                                  dx, dy, dt,
                                  lambda_func=lambda x, y: 0,#.6 if (x>=8 and x<=9 and y <= 7 and y >= 6) else 0,
                                  D_func=lambda x, y: 0 if ((6<=x<=7 and y >=1) or (y<=9 and 4<=x<=5)) else 1,
                                  rho_func=lambda X, Y: np.where((X >= 8) & (X <= 9), 100, 0)
    )
    
    """
    X, Y, rho = solve_rho_eq_2d_cached(start_space_x, end_space_x,
                                  start_space_y, end_space_y,
                                  start_time, end_time,
                                  dx, dy, dt,
                                  lambda_func=lambda x, y: 0,
                                  D_func=lambda x, y: .2,
                                  rho_func=lambda X, Y: np.sin(np.pi * X) * np.sin(np.pi * Y) + 1000
    )
    
    #good rho func np.sin(np.pi * X) * np.sin(np.pi * Y) + 1
    
    X_grid, Y_grid = np.meshgrid(X, Y, indexing='ij')
    
    num_t = rho.shape[0]
    time_indices = np.linspace(0, num_t - 1, 10, dtype=int)
    
    fig = plt.figure(figsize=(20, 10))
    for idx, t_idx in enumerate(time_indices):
        ax = fig.add_subplot(2, 5, idx + 1, projection='3d')
        surf = ax.plot_surface(X_grid, Y_grid, rho[t_idx, :, :], cmap='viridis', edgecolor='none')
        time_val = start_time + t_idx * dt
        ax.set_title(f"Time = {time_val:.2f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('ρ')
        #ax.set_zlim(0, 2)
        fig.colorbar(surf, ax=ax, shrink=.5, aspect=10)
    
    plt.tight_layout()
    plt.show()

def test_diffusion_2d_fullplots():
    start_space_x = 0
    end_space_x = 10.0
    start_space_y = 0
    end_space_y = 10.0
    start_time = 0
    end_time = 40.0

    dx = 0.01
    dy = 0.01
    dt = 0.01

    X, Y, rho = solve_rho_eq_2d_cached(start_space_x, end_space_x,
                                  start_space_y, end_space_y,
                                  start_time, end_time,
                                  dx, dy, dt,
                                  lambda_func=lambda x, y, t: 0,#.2 if (x>=8 and x<=9) else 0,
                                  D_func=lambda x, y, t: 0 if ((6<=x<=7 and y >=1) or (y<=9 and 4<=x<=5)) else 10,
                                  rho_func=lambda X, Y, T: np.where((X >= 8) & (X <= 9), 100, 0)
    )
    
    X_grid, Y_grid = np.meshgrid(X, Y, indexing='ij')
    
    num_t = rho.shape[0]
    time_indices = np.linspace(0, num_t - 1, 10, dtype=int)
    
    for t_idx in time_indices:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X_grid, Y_grid, rho[t_idx, :, :], cmap='viridis', edgecolor='none')
        time_val = start_time + t_idx * dt
        ax.set_title(f"Time = {time_val:.2f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('ρ')
        plt.show() 


def time_test():
    start_space_x = 0
    end_space_x = 10.0
    start_space_y = 0
    end_space_y = 11.0
    start_time = 0
    end_time = 10.0

    #orig .1 all
    dx = 0.1
    dy = 0.1
    dt = 0.05
    
    """
    X, Y, rho = solve_rho_eq_2d(start_space_x, end_space_x,
                                  start_space_y, end_space_y,
                                  start_time, end_time,
                                  dx, dy, dt,
                                  lambda_func=lambda x, y, t: .6 if (x>=8 and x<=9 and y <= 7 and y >= 6) else 0,
                                  D_func=lambda x, y, t: 0 if ((6<=x<=7 and y >=1) or (y<=9 and 4<=x<=5)) else 1,
                                  rho_func=lambda X, Y: np.where((X >= 8) & (X <= 9), 100, 0)
    )
    """
    
    X, Y, rho = solve_rho_eq_2d_cached(start_space_x, end_space_x,
                                  start_space_y, end_space_y,
                                  start_time, end_time,
                                  dx, dy, dt,
                                  lambda_func=lambda x, y, t: .0001,
                                  D_func=lambda x, y, t: .01,
                                  rho_func=lambda X, Y: np.sin(np.pi * X) * np.sin(np.pi * Y) + 1
    )

def plot_total_density_over_time(rho, dt):
    total_density = [np.sum(r) for r in rho]
    times = np.arange(0, len(rho)) * dt
    plt.plot(times, total_density)
    plt.title('Total ρ over Time')
    plt.xlabel('Time')
    plt.ylabel('Total ρ')
    plt.grid(True)
    plt.show()

def heatmap_grid(X, Y, rho, num_plots=9):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    time_indices = np.linspace(0, rho.shape[0] - 1, num_plots, dtype=int)

    for ax, t_idx in zip(axes.flat, time_indices):
        im = ax.imshow(rho[t_idx], origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()], cmap='viridis')
        ax.set_title(f'Time = {t_idx}')
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.tight_layout()
    plt.show()

def contour_plot(X, Y, rho, t_indices):
    X_grid, Y_grid = np.meshgrid(X, Y, indexing='ij')
    for t_idx in t_indices:
        plt.contourf(X_grid, Y_grid, rho[t_idx, :, :], cmap='viridis')
        plt.colorbar()
        plt.title(f'ρ at time = {t_idx}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

def main():
    # Parameters for the simulation
    start_space_x = 0
    end_space_x = 10.0
    start_space_y = 0
    end_space_y = 11.0
    start_time = 0
    end_time = 30.0
    dx = 0.1
    dy = 0.1
    dt = 0.5
    
    # Run the simulation
    X, Y, rho = solve_rho_eq_2d_cached(
        start_space_x, end_space_x,
        start_space_y, end_space_y,
        start_time, end_time,
        dx, dy, dt,
        lambda_func=lambda x, y: 0,
        D_func=lambda x, y: .1,
        rho_func=lambda X, Y: np.sin(np.pi * X) * np.sin(np.pi * Y) + 1
    )

    """
    X, Y, rho = solve_rho_eq_2d_cached(start_space_x, end_space_x,
                                  start_space_y, end_space_y,
                                  start_time, end_time,
                                  dx, dy, dt,
                                  lambda_func=lambda x, y, t: 0,
                                  D_func=lambda x, y: 1,
                                  rho_func=lambda X, Y: np.where((X >= 8) & (X <= 9), 100, 0)
    )
    """
     

    # Visualization calls
    #plot_total_density_over_time(rho, dt)
#(X, Y, rho, num_plots=9)
test_diffusion_2d()


if __name__ == "__main__":
    main()
