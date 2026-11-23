"""Route (column) enumeration for the set-partitioning formulation.

A column ``r = (o, v, S, pi)`` is a trip of vehicle type ``v`` from depot ``o``
visiting the facility set ``S`` (|S| <= L) in order ``pi``. By default only the
distance-minimising order of each set is kept (``all_orders=False``); with
load-dependent energy the cost-optimal order can differ, so ``all_orders=True``
keeps every permutation and makes the model exact over all elementary routes
with at most L stops.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

from ..data.model import LongHaulInstance
from .costs import TripCostModel, trip_cost_model


@dataclass(frozen=True)
class Column:
    idx: int
    depot: str
    vehicle: str
    stops: tuple
    model: TripCostModel
    max_trips: int


def _useful_facilities(inst: LongHaulInstance, depot_id: str) -> list:
    stock = inst.D[depot_id].stock
    return [f for f in inst.facilities if any(stock.get(w, 0) > 0 for w in f.fees)]


def best_order(inst: LongHaulInstance, depot: str, stops) -> tuple:
    best, best_d = None, float("inf")
    for perm in permutations(stops):
        path = (depot, *perm, depot)
        d = sum(inst.d(a, b) for a, b in zip(path[:-1], path[1:]))
        if d < best_d - 1e-9:
            best, best_d = perm, d
    return tuple(best)


def enumerate_columns(inst: LongHaulInstance, max_stops: int | None = None,
                      all_orders: bool = False, vehicles=None) -> list[Column]:
    L = max_stops or inst.params.max_stops
    vehicles = vehicles or inst.vehicles
    cols: list[Column] = []
    for dp in inst.depots:
        facs = [f.id for f in _useful_facilities(inst, dp.id)]
        for k in range(1, L + 1):
            for S in combinations(facs, k):
                orders = permutations(S) if all_orders else [best_order(inst, dp.id, S)]
                for order in orders:
                    for v in vehicles:
                        m = trip_cost_model(inst, dp.id, v, tuple(order))
                        if not m.feasible_range:
                            continue
                        ub = int(v.vehicle_days // m.time.days)
                        if ub <= 0:
                            continue
                        cols.append(Column(len(cols), dp.id, v.id, tuple(order), m, ub))
    return cols
