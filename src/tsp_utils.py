"""Utilidades compartidas para TSP: instancias, 2-opt, vecino más cercano, MST, óptimo exacto y SA."""
import itertools
import math
import random
import numpy as np


def random_cities(n, rng):
    return rng.random((n, 2)) * 100


def dist_matrix(pts):
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def tour_cost(tour, D):
    t = np.asarray(tour)
    return float(D[t, np.roll(t, -1)].sum())


def nearest_neighbor(D, start):
    n = len(D)
    visited = [start]
    left = set(range(n)) - {start}
    while left:
        last = visited[-1]
        nxt = min(left, key=lambda j: D[last, j])
        visited.append(nxt)
        left.remove(nxt)
    return np.array(visited)


def two_opt(tour, D):
    """2-opt (primera mejora). Devuelve (tour, número de mejoras)."""
    t = np.array(tour)
    n = len(t)
    improvements = 0
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = t[i], t[i + 1]
                c, d = t[j], t[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -1e-12:
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1].copy()
                    improvements += 1
                    improved = True
    return t, improvements


def mst_approximation(D):
    """2-aproximación: MST (Prim) + recorrido preorden + atajos."""
    n = len(D)
    in_tree = [False] * n
    parent = [-1] * n
    key = [math.inf] * n
    key[0] = 0
    children = {i: [] for i in range(n)}
    for _ in range(n):
        u = min((i for i in range(n) if not in_tree[i]), key=lambda i: key[i])
        in_tree[u] = True
        if parent[u] >= 0:
            children[parent[u]].append(u)
        for v in range(n):
            if not in_tree[v] and D[u, v] < key[v]:
                key[v] = D[u, v]
                parent[v] = u
    order = []
    stack = [0]
    while stack:
        u = stack.pop()
        order.append(u)
        stack.extend(reversed(children[u]))
    return np.array(order)


def optimal_tsp(D):
    """Óptimo exacto por fuerza bruta (válido para ~9-10 ciudades)."""
    n = len(D)
    perms = np.array(list(itertools.permutations(range(1, n))))
    first = np.zeros((len(perms), 1), dtype=int)
    tours = np.hstack([first, perms])
    nxt = np.roll(tours, -1, axis=1)
    costs = D[tours, nxt].sum(axis=1)
    k = costs.argmin()
    return tours[k], float(costs[k])


def temperature(schedule, k, N, T0, Tf):
    x = k / N
    if schedule == "lineal":
        return T0 + (Tf - T0) * x
    if schedule == "exponencial":
        return T0 * (Tf / T0) ** x
    if schedule == "logarítmico":
        # T0 / (1 + a*ln(1+k)) con 'a' calibrada para terminar en Tf
        a = (T0 / Tf - 1) / math.log(1 + N)
        return T0 / (1 + a * math.log(1 + k))
    raise ValueError(schedule)


def simulated_annealing(D, schedule="exponencial", iters=50_000, T0=None, Tf=0.01,
                        seed=0, record_every=500):
    """Recocido simulado con 2-opt como movimiento, inicio aleatorio."""
    rnd = random.Random(seed)
    n = len(D)
    tour = list(range(n))
    rnd.shuffle(tour)
    cost = tour_cost(tour, D)
    if T0 is None:
        T0 = cost / n  # escala razonable según tamaño de la instancia
    best, best_cost = tour[:], cost
    history = []
    for k in range(iters):
        T = temperature(schedule, k, iters, T0, Tf)
        i, j = sorted(rnd.sample(range(n), 2))
        if j - i < 1 or (i == 0 and j == n - 1):
            continue
        a, b = tour[i - 1], tour[i]
        c, d = tour[j], tour[(j + 1) % n]
        delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
        if delta < 0 or rnd.random() < math.exp(-delta / T):
            tour[i:j + 1] = tour[i:j + 1][::-1]
            cost += delta
            if cost < best_cost - 1e-12:
                best, best_cost = tour[:], cost
        if k % record_every == 0:
            history.append((k, T, best_cost))
    return np.array(best), best_cost, history
