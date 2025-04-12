from time import perf_counter
import MultiDimCached
import MultiDimRho
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
MultiDimRho.time_test()
end_original = perf_counter()

# Time the cached method
start_cached = perf_counter()
MultiDimCached.time_test()
end_cached = perf_counter()

# Calculate the execution times
time_original = end_original - start_original
time_cached = end_cached - start_cached

print(f"Uncached time: {time_original}\nCached time: {time_cached}")
