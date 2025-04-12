import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import MatrixOperations as mat

def solve_rho_eq_2d(start_space_x=0, end_space_x=10.0,
                      start_space_y=0, end_space_y=10.0,
                      start_time=0, end_time=10.0,
                      dx=0.1, dy=0.1, dt=0.1,
                      lambda_func=lambda x, y, t: 0,
                      D_func=lambda x, y, t: 1,
                      rho_func=lambda X, Y: np.sin(np.pi * X) * np.sin(np.pi * Y) + 1):
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
    
    # Time-stepping loop
    for n in range(num_t - 1):
        # x-sweep: For each fixed y, solve the 1D implicit system in x
        rho_temp = np.zeros((num_x, num_y), dtype="float64")
        for j in range(num_y):
            a = np.zeros(num_x, dtype="float64")
            b = np.zeros(num_x, dtype="float64")
            c = np.zeros(num_x, dtype="float64")
            d = np.zeros(num_x, dtype="float64")
            
            # Left boundary
            D_ip = (D_func(X[0], Y[j], T[n]) + D_func(X[1], Y[j], T[n])) / 2.0
            b[0] = 1 + (dt / dx**2) * D_ip
            c[0] = - dt * D_ip / dx**2
            d[0] = rho[n, 0, j] * (1 - dt * lambda_func(X[0], Y[j], T[n]))
            
            # Internal points
            for i in range(1, num_x - 1):
                D_ip = (D_func(X[i], Y[j], T[n]) + D_func(X[i+1], Y[j], T[n])) / 2.0
                D_im = (D_func(X[i], Y[j], T[n]) + D_func(X[i-1], Y[j], T[n])) / 2.0
                a[i] = - dt * D_im / dx**2
                b[i] = 1 + dt / dx**2 * (D_ip + D_im)
                c[i] = - dt * D_ip / dx**2
                d[i] = rho[n, i, j] * (1 - dt * lambda_func(X[i], Y[j], T[n]))
            
            # Right boundary
            D_im = (D_func(X[num_x-1], Y[j], T[n]) + D_func(X[num_x-2], Y[j], T[n])) / 2.0
            a[-1] = - dt * D_im / dx**2 
            b[-1] = 1 + dt * D_im / dx**2
            d[-1] = rho[n, num_x-1, j] * (1 - dt * lambda_func(X[num_x-1], Y[j], T[n]))
            
            rho_temp[:, j] = mat.solve_matrix_thomas(a, b, c, d)
        # y-sweep: For each fixed x in y
        rho_new = np.zeros((num_x, num_y), dtype="float64")
        for i in range(num_x):
            a = np.zeros(num_y, dtype="float64")
            b = np.zeros(num_y, dtype="float64")
            c = np.zeros(num_y, dtype="float64")
            d = np.zeros(num_y, dtype="float64")
            
            # Bottom boundary
            D_jp = (D_func(X[i], Y[0], T[n]) + D_func(X[i], Y[1], T[n])) / 2.0
            b[0] = 1 + (dt / dy**2) * D_jp
            c[0] = - dt * D_jp / dy**2
            d[0] = rho_temp[i, 0] * (1 - dt * lambda_func(X[i], Y[0], T[n]))
            
            # Internal points
            for j in range(1, num_y - 1):
                D_jp = (D_func(X[i], Y[j], T[n]) + D_func(X[i], Y[j+1], T[n])) / 2.0
                D_jm = (D_func(X[i], Y[j], T[n]) + D_func(X[i], Y[j-1], T[n])) / 2.0
                a[j] = - dt * D_jm / dy**2
                b[j] = 1 + dt / dy**2 * (D_jp + D_jm)
                c[j] = - dt * D_jp / dy**2
                d[j] = rho_temp[i, j] * (1 - dt * lambda_func(X[i], Y[j], T[n]))
            
            # Top boundary
            D_jm = (D_func(X[i], Y[num_y-1], T[n]) + D_func(X[i], Y[num_y-2], T[n])) / 2.0
            a[-1] = - dt * D_jm / dy**2
            b[-1] = 1 + dt * D_jm / dy**2
            d[-1] = rho_temp[i, num_y-1] * (1 - dt * lambda_func(X[i], Y[num_y-1], T[n]))
            
            rho_new[i, :] = mat.solve_matrix_thomas(a, b, c, d)
        
        # Update the solution for the next time step
        rho[n+1, :, :] = rho_new.copy()
    
    return X, Y, rho

def test_diffusion_2d():
    start_space_x = 0
    end_space_x = 10.0
    start_space_y = 0
    end_space_y = 10.0
    start_time = 0
    end_time = 30.0

    #orig .1 all
    dx = 0.1
    dy = 0.1
    dt = 0.1  
    
    
    X, Y, rho = solve_rho_eq_2d(start_space_x, end_space_x,
                                  start_space_y, end_space_y,
                                  start_time, end_time,
                                  dx, dy, dt,
                                  lambda_func=lambda x, y, t: 0,
                                  D_func=lambda x, y, t: .2,
                                  rho_func=lambda X, Y: np.sin(np.pi * X) * np.sin(np.pi * Y) + 10
    )
    """    
    X, Y, rho = solve_rho_eq_2d(start_space_x, end_space_x,
                                  start_space_y, end_space_y,
                                  start_time, end_time,
                                  dx, dy, dt,
                                  lambda_func=lambda x, y,t : 0,#.6 if (x>=8 and x<=9 and y <= 7 and y >= 6) else 0,
                                  D_func=lambda x, y, t: 0 if ((6<=x<=7 and y >=1) or (y<=9 and 4<=x<=5)) else 1,
                                  rho_func=lambda X, Y: np.where((X >= 8) & (X <= 9), 100, 0))

    #good rho func np.sin(np.pi * X) * np.sin(np.pi * Y) + 1
    """
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
        #ax.set_zlim(0, 21)
        fig.colorbar(surf, ax=ax, shrink=.5, aspect=10)
    
    plt.tight_layout()
    plt.show()

def test_diffusion_2d_fullplots():
    start_space_x = 0
    end_space_x = 10.0
    start_space_y = 0
    end_space_y = 10.0
    start_time = 0
    end_time = 80.0

    dx = 0.1
    dy = 0.1
    dt = 0.1

    X, Y, rho = solve_rho_eq_2d(start_space_x, end_space_x,
                                  start_space_y, end_space_y,
                                  start_time, end_time,
                                  dx, dy, dt,
                                  lambda_func=lambda x, y, t: 0,#.2 if (x>=8 and x<=9) else 0,
                                  D_func=lambda x, y, t: 0 if ((6<=x<=7 and y >=1) or (y<=9 and 4<=x<=5)) else 10,
                                  rho_func=lambda X, Y: np.where((X >= 8) & (X <= 9), 100, 0)
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
    

def main():
    test_diffusion_2d()

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
    
    X, Y, rho = solve_rho_eq_2d(start_space_x, end_space_x,
                                  start_space_y, end_space_y,
                                  start_time, end_time,
                                  dx, dy, dt,
                                  lambda_func=lambda x, y, t: .0001,
                                  D_func=lambda x, y, t: .01,
                                  rho_func=lambda X, Y: np.sin(np.pi * X) * np.sin(np.pi * Y) + 1
    )

if __name__ == "__main__":
    main()
