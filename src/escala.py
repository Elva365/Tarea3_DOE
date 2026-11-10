import numpy as np
from tsp_utils import *

for n in [5, 10, 15]:
    rng = np.random.default_rng(7 + n)
    D = dist_matrix(random_cities(n, rng))
    d_prom = D[np.triu_indices(n, 1)].mean()
    print(f"n={n}: distancia promedio entre ciudades = {d_prom:.3f}")