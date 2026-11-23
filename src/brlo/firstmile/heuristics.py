"""Classical CVRP heuristics used as baselines for the learned policy.

* nearest-neighbour construction,
* Clarke & Wright (1964) parallel savings,
* local search: 2-opt (intra-route), relocate and swap (inter-route), first
  improvement until a local optimum (or an iteration cap).

A solution is a list of routes, each a list of customer indices (1..n);
the depot (0) is implicit at both ends.
"""

from __future__ import annotations

import numpy as np


def dist_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def route_length(r, D) -> float:
    if not r:
        return 0.0
    return D[0, r[0]] + sum(D[a, b] for a, b in zip(r[:-1], r[1:])) + D[r[-1], 0]


def total_length(routes, D) -> float:
    return sum(route_length(r, D) for r in routes)


def check(routes, demand, cap) -> None:
    seen = sorted(c for r in routes for c in r)
    assert seen == list(range(1, len(demand))), "every customer exactly once"
    for r in routes:
        assert sum(demand[c] for c in r) <= cap, "capacity"


def nearest_neighbour(coords, demand, cap):
    D = dist_matrix(coords)
    n = len(demand) - 1
    left = set(range(1, n + 1))
    routes = []
    while left:
        r, load, cur = [], 0, 0
        while True:
            cand = [c for c in left if load + demand[c] <= cap]
            if not cand:
                break
            nxt = min(cand, key=lambda c: D[cur, c])
            r.append(nxt); load += demand[nxt]; left.remove(nxt); cur = nxt
        routes.append(r)
    return routes


def clarke_wright(coords, demand, cap):
    D = dist_matrix(coords)
    n = len(demand) - 1
    routes = {i: [i] for i in range(1, n + 1)}
    where = {i: i for i in range(1, n + 1)}
    load = {i: demand[i] for i in range(1, n + 1)}
    i_idx, j_idx = np.triu_indices(n, 1)
    i_idx, j_idx = i_idx + 1, j_idx + 1
    sav = D[0, i_idx] + D[0, j_idx] - D[i_idx, j_idx]
    for k in np.argsort(-sav):
        if sav[k] <= 0:
            break
        i, j = int(i_idx[k]), int(j_idx[k])
        ri, rj = where[i], where[j]
        if ri == rj or load[ri] + load[rj] > cap:
            continue
        a, b = routes[ri], routes[rj]
        # i and j must be route ends; orient so that a ends with i and b starts with j
        if a[-1] != i:
            if a[0] == i:
                a = a[::-1]
            else:
                continue
        if b[0] != j:
            if b[-1] == j:
                b = b[::-1]
            else:
                continue
        merged = a + b
        routes[ri] = merged
        load[ri] += load[rj]
        del routes[rj], load[rj]
        for c in b:
            where[c] = ri
    return list(routes.values())


def _two_opt(r, D):
    improved = True
    while improved:
        improved = False
        path = [0] + r + [0]
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                delta = (D[path[i - 1], path[j]] + D[path[i], path[j + 1]]
                         - D[path[i - 1], path[i]] - D[path[j], path[j + 1]])
                if delta < -1e-10:
                    path[i:j + 1] = path[i:j + 1][::-1]
                    improved = True
        r = path[1:-1]
    return r


def local_search(routes, coords, demand, cap, max_rounds: int = 50):
    D = dist_matrix(coords)
    routes = [list(r) for r in routes if r]
    loads = [sum(demand[c] for c in r) for r in routes]
    for _ in range(max_rounds):
        routes = [_two_opt(r, D) for r in routes]
        moved = False
        # relocate: move customer c from route a (pos p) to route b (best position)
        for a in range(len(routes)):
            p = 0
            while p < len(routes[a]):
                c = routes[a][p]
                ra = routes[a]
                prev = ra[p - 1] if p > 0 else 0
                nxt = ra[p + 1] if p + 1 < len(ra) else 0
                gain = D[prev, c] + D[c, nxt] - D[prev, nxt]
                best = (1e-10, None, None)
                for b in range(len(routes)):
                    if b == a or loads[b] + demand[c] > cap:
                        continue
                    path = [0] + routes[b] + [0]
                    for q in range(len(path) - 1):
                        add = D[path[q], c] + D[c, path[q + 1]] - D[path[q], path[q + 1]]
                        if gain - add > best[0]:
                            best = (gain - add, b, q)
                if best[1] is not None:
                    _, b, q = best
                    routes[a].pop(p); loads[a] -= demand[c]
                    routes[b].insert(q, c); loads[b] += demand[c]
                    moved = True
                else:
                    p += 1
        # swap customers between routes
        for a in range(len(routes)):
            for b in range(a + 1, len(routes)):
                done = False
                for p, c in enumerate(routes[a]):
                    for q, e in enumerate(routes[b]):
                        if loads[a] - demand[c] + demand[e] > cap or loads[b] - demand[e] + demand[c] > cap:
                            continue
                        ra, rb = routes[a], routes[b]
                        pa, na = (ra[p - 1] if p else 0), (ra[p + 1] if p + 1 < len(ra) else 0)
                        pb, nb = (rb[q - 1] if q else 0), (rb[q + 1] if q + 1 < len(rb) else 0)
                        delta = (D[pa, e] + D[e, na] - D[pa, c] - D[c, na]
                                 + D[pb, c] + D[c, nb] - D[pb, e] - D[e, nb])
                        if delta < -1e-10:
                            ra[p], rb[q] = e, c
                            loads[a] += demand[e] - demand[c]
                            loads[b] += demand[c] - demand[e]
                            moved = done = True
                            break
                    if done:
                        break
        keep = [i for i, r in enumerate(routes) if r]
        routes = [routes[i] for i in keep]
        loads = [loads[i] for i in keep]
        if not moved:
            break
    return [_two_opt(r, D) for r in routes]


def solve_classical(coords, demand, cap, method: str = "cw+ls"):
    if method == "nn":
        return nearest_neighbour(coords, demand, cap)
    if method == "cw":
        return clarke_wright(coords, demand, cap)
    if method == "cw+ls":
        return local_search(clarke_wright(coords, demand, cap), coords, demand, cap)
    raise ValueError(method)
