"""First-mile collection instances (CVRP).

Pick-up points are workshops, car dealers and dismantlers that hand over
end-of-life batteries; the collector returns to the regional storage depot.

Two generators:

* ``uniform``   - the standard learning-to-route benchmark distribution
  (Nazari et al., 2018; Kool et al., 2019): coordinates U[0,1]^2, integer
  demands U{1..9}, capacity 30/40/50 for n = 20/50/100. Kept for comparability
  with published neural CVRP results.
* ``clustered`` - realistic spatial structure: a mixture of town clusters
  (Gaussian, random spread) plus a uniform rural background, the depot placed
  off-centre (industrial estate), and heavier, rarer pick-ups (EV packs) mixed
  with frequent light ones (portable batteries).

Coordinates are normalised to [0,1]^2; :func:`to_km` rescales a solution to
a region of a given side length (Euclidean x circuity) for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAPACITY = {10: 20, 20: 30, 50: 40, 100: 50, 200: 50}


@dataclass
class CVRPBatch:
    coords: np.ndarray     # [B, N+1, 2], node 0 = depot
    demand: np.ndarray     # [B, N+1] integer units, demand[:, 0] = 0
    capacity: int

    @property
    def n(self) -> int:
        return self.coords.shape[1] - 1


def uniform(batch: int, n: int, seed: int | None = None) -> CVRPBatch:
    rng = np.random.default_rng(seed)
    coords = rng.random((batch, n + 1, 2))
    demand = rng.integers(1, 10, size=(batch, n + 1))
    demand[:, 0] = 0
    return CVRPBatch(coords, demand, CAPACITY.get(n, 50))


def clustered(batch: int, n: int, seed: int | None = None, n_towns=(3, 8), rural=0.25) -> CVRPBatch:
    rng = np.random.default_rng(seed)
    coords = np.empty((batch, n + 1, 2))
    demand = np.empty((batch, n + 1), dtype=int)
    for b in range(batch):
        k = rng.integers(n_towns[0], n_towns[1] + 1)
        centres = rng.random((k, 2)) * 0.8 + 0.1
        spread = rng.uniform(0.02, 0.08, size=k)
        weight = rng.dirichlet(np.ones(k) * 0.8)
        n_rural = rng.binomial(n, rural)
        town = rng.choice(k, size=n - n_rural, p=weight)
        pts = centres[town] + rng.normal(size=(n - n_rural, 2)) * spread[town, None]
        pts = np.concatenate([pts, rng.random((n_rural, 2))])
        coords[b, 1:] = np.clip(pts, 0, 1)
        coords[b, 0] = centres[0] + rng.normal(size=2) * 0.1          # industrial estate near a town
        coords[b, 0] = np.clip(coords[b, 0], 0, 1)
        heavy = rng.random(n) < 0.12                                   # EV packs / modules
        demand[b, 1:] = np.where(heavy, rng.integers(6, 10, size=n), rng.integers(1, 4, size=n))
        demand[b, 0] = 0
    return CVRPBatch(coords, demand, CAPACITY.get(n, 50))


def to_km(length_unit: float, side_km: float = 120.0, circuity: float = 1.3) -> float:
    return length_unit * side_km * circuity
