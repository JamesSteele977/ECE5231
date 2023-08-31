import numpy as np

def generate_nd_grid(domains, sampling_rates):
    # Create slice objects for each dimension based on the domains and sampling rates
    slice_objects = [slice(start, stop, 1j * rate) for (start, stop), rate in zip(domains, sampling_rates)]

    # Use np.ogrid to generate the grid; this uses broadcasting and doesn't actually create a full n-dimensional grid in memory
    grid_arrays = np.ogrid[slice_objects]

    # Stack these arrays along the last axis to get the final grid
    grid = np.stack(grid_arrays, axis=-1)

    return grid


domain = generate_nd_grid(np.array([[0, 10], [0, 10]]), np.array([11, 11]))

print(domain)