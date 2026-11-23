"""Solution representation and the independent evaluator.

Every optimiser in this package (legacy heuristic, MILP, ALNS) outputs a
:class:`Plan`; :func:`evaluate` recomputes its cost from scratch and lists
every violated constraint. Results reported in the paper always come from
this function, never from an optimiser's own objective value.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..data.model import LongHaulInstance
from .costs import leg_loads, trip_cost_model

TOL = 1e-6


@dataclass
class Trip:
    depot: str
    vehicle: str
    stops: tuple
    deliveries: dict = field(default_factory=dict)   # (stop_index, waste) -> kg
    containers: dict = field(default_factory=dict)   # (stop_index, waste) -> count

    def cargo_kg(self) -> float:
        return sum(self.deliveries.values())


@dataclass
class Plan:
    trips: list = field(default_factory=list)
    unshipped: dict = field(default_factory=dict)    # (depot, waste) -> kg
    meta: dict = field(default_factory=dict)


@dataclass
class Evaluation:
    cost: float
    transport: float
    gate_fees: float
    penalty: float
    co2_kg: float
    distance_km: float
    n_trips: int
    shipped_kg: float
    unshipped_kg: float
    violations: list

    @property
    def feasible(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("cost", "transport", "gate_fees", "penalty", "co2_kg",
                                           "distance_km", "n_trips", "shipped_kg", "unshipped_kg")}
        d["feasible"] = self.feasible
        d["n_violations"] = len(self.violations)
        return d


def evaluate(inst: LongHaulInstance, plan: Plan) -> Evaluation:
    p = inst.params
    viol: list[str] = []
    transport = fees = co2 = dist = 0.0
    shipped = defaultdict(float)          # (depot, waste)
    received = defaultdict(float)         # (facility, waste)
    vdays = defaultdict(float)
    incompatible = {frozenset(x) for x in inst.incompatible}

    for t_idx, t in enumerate(plan.trips):
        tag = f"trip{t_idx}[{t.depot}/{t.vehicle}/{'-'.join(t.stops)}]"
        v = inst.V[t.vehicle]
        if len(t.stops) > p.max_stops:
            viol.append(f"{tag}: {len(t.stops)} stops > max_stops")
        if len(set(t.stops)) != len(t.stops):
            viol.append(f"{tag}: repeated stop")
        m = trip_cost_model(inst, t.depot, v, tuple(t.stops))
        if not m.feasible_range:
            viol.append(f"{tag}: exceeds electric daily range")
        tare = slots = 0.0
        wastes_on_board = set()
        for (i, w), q in t.deliveries.items():
            if q < -TOL:
                viol.append(f"{tag}: negative delivery")
            if q <= TOL:
                continue
            fac = inst.F[t.stops[i]]
            if w not in fac.fees:
                viol.append(f"{tag}: {fac.id} does not accept {w}")
                continue
            n = t.containers.get((i, w), 0)
            c = inst.container_of(w)
            if q > n * c.net_kg + TOL:
                viol.append(f"{tag}: {q:.0f} kg of {w} in {n} x {c.id} (cap {c.net_kg})")
            shipped[(t.depot, w)] += q
            received[(fac.id, w)] += q
            fees += fac.fees[w] * q
            wastes_on_board.add(w)
        for (i, w), n in t.containers.items():
            c = inst.container_of(w)
            tare += n * c.tare_kg
            slots += n * c.slots
        if t.cargo_kg() + tare > v.payload_kg + TOL:
            viol.append(f"{tag}: overload {t.cargo_kg() + tare:.0f} > {v.payload_kg:.0f} kg")
        if slots > v.slots + TOL:
            viol.append(f"{tag}: {slots:.1f} slots > {v.slots}")
        for a in wastes_on_board:
            for b in wastes_on_board:
                if a < b and frozenset((a, b)) in incompatible:
                    viol.append(f"{tag}: {a} and {b} may not share a vehicle")
        loads = leg_loads(tuple(t.stops), t.deliveries, tare)
        transport += m.fixed + sum(b * L for b, L in zip(m.beta, loads))
        co2 += m.co2_fixed + sum(g * L for g, L in zip(m.gamma, loads))
        dist += m.distance_km
        vdays[t.vehicle] += m.time.days

    for vid, used in vdays.items():
        if used > inst.V[vid].vehicle_days + TOL:
            viol.append(f"fleet {vid}: {used} vehicle-days > {inst.V[vid].vehicle_days}")

    unshipped_total = 0.0
    for dp in inst.depots:
        for w, s in dp.stock.items():
            u = plan.unshipped.get((dp.id, w), 0.0)
            if shipped[(dp.id, w)] + u > s + 1e-3 * max(1.0, s) or shipped[(dp.id, w)] + u < s - 1e-3 * max(1.0, s):
                viol.append(f"balance {dp.id}/{w}: shipped {shipped[(dp.id, w)]:.0f} + unshipped {u:.0f} != stock {s:.0f}")
            unshipped_total += max(0.0, s - shipped[(dp.id, w)])
    for f in inst.facilities:
        tot = 0.0
        for w in f.fees:
            r = received[(f.id, w)]
            tot += r
            if r > f.cap.get(w, float("inf")) + 1e-3:
                viol.append(f"capacity {f.id}/{w}: {r:.0f} > {f.cap[w]:.0f}")
            ml = f.min_lot.get(w, 0.0)
            if TOL < r < ml - 1e-3:
                viol.append(f"min lot {f.id}/{w}: {r:.0f} < {ml:.0f}")
        if tot > f.total_cap + 1e-3:
            viol.append(f"total capacity {f.id}: {tot:.0f} > {f.total_cap:.0f}")

    penalty = p.unshipped_penalty_eur_kg * unshipped_total
    co2_cost = 0.0  # already inside transport via the CO2 shadow price
    return Evaluation(cost=transport + fees + penalty + co2_cost, transport=transport, gate_fees=fees,
                      penalty=penalty, co2_kg=co2, distance_km=dist, n_trips=len(plan.trips),
                      shipped_kg=sum(shipped.values()), unshipped_kg=unshipped_total, violations=viol)
