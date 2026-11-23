"""Cost model for a single long-haul trip.

A trip is ``depot -> s_1 -> ... -> s_m -> depot``. Its cost decomposes into a
load-independent part (fixed cost, distance-based cost, driver time, nights
away, energy at zero load) and a part that is *linear* in the mass carried on
every leg (incremental energy). Because mass on leg ``l`` is the sum of the
deliveries made after that leg plus the container tare, the trip cost is a
linear function of the delivery variables. This is what lets the set
partitioning MILP in :mod:`brlo.longhaul.milp` price fuel and CO2 exactly
while keeping the formulation linear.

Energy model (per vehicle, per 100 km): e(L) = e_empty + (e_full - e_empty) * L / P
(linear load-dependent fuel consumption, cf. Xiao et al., 2012; Bektas & Laporte, 2011).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..data.model import LongHaulInstance, Params, VehicleType


@dataclass(frozen=True)
class TripTime:
    driving_h: float
    service_h: float
    breaks_h: float
    rests: int           # number of daily rests (= nights away)
    elapsed_h: float     # wall-clock duration
    days: int            # calendar vehicle-days occupied


def hos_schedule(leg_driving_h: list[float], service_h: list[float], p: Params) -> TripTime:
    """Simulate a single-driver schedule under EU Regulation (EC) 561/2006.

    Rules modelled: a 45-min break after 4.5 h of cumulative driving, at most
    9 h of driving per daily duty period, followed by an 11 h daily rest.
    ``service_h[i]`` is the (un)loading time after leg ``i`` (other work: it
    neither counts as driving nor as a break). Driving is split across
    breaks/rests whenever a limit is reached mid-leg.
    """
    drive_since_break = 0.0
    drive_today = 0.0
    breaks = 0.0
    rests = 0
    elapsed = 0.0
    for i, leg in enumerate(leg_driving_h):
        remaining = leg
        while remaining > 1e-9:
            room = min(p.max_continuous_driving_h - drive_since_break,
                       p.max_daily_driving_h - drive_today)
            if room <= 1e-9:
                if drive_today >= p.max_daily_driving_h - 1e-9:
                    rests += 1
                    elapsed += p.daily_rest_h
                    drive_today = 0.0
                else:
                    breaks += p.break_h
                    elapsed += p.break_h
                drive_since_break = 0.0
                continue
            step = min(room, remaining)
            remaining -= step
            drive_since_break += step
            drive_today += step
            elapsed += step
        if i < len(service_h):
            elapsed += service_h[i]
    driving = sum(leg_driving_h)
    service = sum(service_h)
    days = max(1, math.ceil(elapsed / 24.0))
    return TripTime(driving, service, breaks, rests, elapsed, days)


@dataclass(frozen=True)
class TripCostModel:
    """Everything needed to price one trip of vehicle ``v`` on a fixed stop
    sequence. ``fixed`` is the per-trip cost at zero load, ``beta[l]`` the
    marginal EUR/kg carried on leg ``l`` and ``gamma[l]`` the marginal kg CO2
    per kg carried on leg ``l``."""

    stops: tuple
    depot: str
    vehicle: str
    legs_km: tuple
    distance_km: float
    time: TripTime
    fixed: float
    co2_fixed: float
    beta: tuple
    gamma: tuple
    feasible_range: bool


def energy_price_and_co2(v: VehicleType, p: Params) -> tuple[float, float]:
    if v.energy == "diesel":
        return p.diesel_eur_l, p.co2_kg_per_l_diesel
    if v.energy == "electric":
        return p.electricity_eur_kwh, p.co2_kg_per_kwh
    raise ValueError(f"unknown energy carrier {v.energy}")


def trip_cost_model(inst: LongHaulInstance, depot: str, v: VehicleType, stops: tuple) -> TripCostModel:
    p = inst.params
    path = (depot, *stops, depot)
    legs = tuple(inst.d(a, b) for a, b in zip(path[:-1], path[1:]))
    dist = sum(legs)
    drive = [km / v.speed_kmh for km in legs]
    serv = [inst.F[s].service_h for s in stops]
    t = hos_schedule(drive, serv, p)

    price, co2_per_unit = energy_price_and_co2(v, p)
    e0 = v.e_empty / 100.0                                   # units per km, empty
    de = (v.e_full - v.e_empty) / 100.0 / v.payload_kg       # units per km per kg
    co2_price = p.co2_price_eur_t / 1000.0

    labour = p.driver_eur_h * (t.driving_h + t.service_h + t.breaks_h)
    nights = p.per_diem_eur * t.rests
    energy0 = e0 * dist
    fixed = (v.fixed_cost + v.cost_per_km * dist + labour + nights
             + energy0 * price + energy0 * co2_per_unit * co2_price)
    beta = tuple(km * de * (price + co2_per_unit * co2_price) for km in legs)
    gamma = tuple(km * de * co2_per_unit for km in legs)

    ok = True
    if v.daily_range_km is not None:
        # Range check per duty day: the distance driven in any day cannot exceed
        # the daily range (approximated as total distance / days).
        ok = dist / max(1, t.rests + 1) <= v.daily_range_km + 1e-6 and max(legs) <= v.daily_range_km + 1e-6
    return TripCostModel(stops, depot, v.id, legs, dist, t, fixed, energy0 * co2_per_unit, beta, gamma, ok)


def leg_loads(stops: tuple, deliveries: dict, tare: float) -> list[float]:
    """Mass carried on each leg given ``deliveries[(stop_index, waste)] = kg``.
    Containers (tare) stay on board for the whole trip (empties come back)."""
    m = len(stops)
    per_stop = [0.0] * m
    for (i, _w), q in deliveries.items():
        per_stop[i] += q
    loads = []
    for l in range(m + 1):          # leg l arrives at stop l (or depot when l == m)
        loads.append(sum(per_stop[l:]) + tare)
    return loads
