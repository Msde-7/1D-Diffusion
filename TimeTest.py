from time import perf_counter
import Cached
import RhoEq
import numpy as np

# Dummy implementations for original and cached methods
start_space = 0
end_space=10.0
start_time = 0
end_time = 10.0
dx = .001
dt = .1
    
num_x = round((end_space - start_space) / dx)
num_t = round((end_time - start_time) / dt)

    
# Time the original method
start_original = perf_counter()
RhoEq.solve_rho_eq(start_space = start_space, end_space = end_space, start_time = start_time, end_time = end_time, dx = dx, dt = dt, lambda_func=lambda x, t: 0, D_func=lambda x, t: .1 * x, rho_func=lambda X: np.sin(np.pi * X) + 1)
end_original = perf_counter()

# Time the cached method
start_cached = perf_counter()
Cached.solve_rho_eq(start_space = start_space, end_space = end_space, start_time = start_time, end_time = end_time, dx = dx, dt = dt, lambda_func=lambda x: 0, D_func=lambda x: .1 * x, rho_func=lambda X: np.sin(np.pi * X) + 1)
end_cached = perf_counter()

# Calculate the execution times
time_original = end_original - start_original
time_cached = end_cached - start_cached

print(f"Uncached time: {time_original}\nCached time: {time_cached}");
