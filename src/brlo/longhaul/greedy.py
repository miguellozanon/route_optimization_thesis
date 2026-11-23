"""A sensible constructive heuristic (what a careful planner would do by hand).

1. Price each (stream, facility) pair by gate fee + an estimate of the
   full-truckload transport cost per kg (best vehicle, single-stop trip).
2. Assign stock stream by stream to the cheapest facilities, honouring
   per-stream and facility-wide capacities and minimum lots.
3. Consolidate per (depot, facility): full truckloads go out as direct trips;
   the residual partial loads are merged by a nearest-neighbour sweep into
   multi-stop trips of up to ``max_stops`` stops.

Used as a stronger baseline than the legacy prototype and as the initial
solution of the ALNS matheuristic.
"""

from __future__ import annotations

import math
from collections import defaultdict

from ..data.model import LongHaulInstance
from .costs import trip_cost_model
from .solution import Plan, Trip


def _per_kg_transport(inst, depot, fac_id):
    best = math.inf
    for v in inst.vehicles:
        m = trip_cost_model(inst, depot, v, (fac_id,))
        if m.feasible_range:
            best = min(best, (m.fixed + sum(m.beta) * v.payload_kg * 0.8) / (v.payload_kg * 0.8))
    return best


def greedy_plan(inst: LongHaulInstance) -> Plan:
    p = inst.params
    used_k = defaultdict(float)
    used_j = defaultdict(float)
    alloc = defaultdict(float)              # (depot, facility, stream) -> kg
    unshipped = {}
    for dp in inst.depots:
        tr = {f.id: _per_kg_transport(inst, dp.id, f.id) for f in inst.facilities}
        for k, s in sorted(dp.stock.items(), key=lambda kv: -kv[1]):
            rem = s
            facs = sorted((f for f in inst.facilities if k in f.fees), key=lambda f: f.fees[k] + tr[f.id])
            for f in facs:
                if rem <= 1e-9:
                    break
                if f.fees[k] + tr[f.id] > p.unshipped_penalty_eur_kg:
                    break
                room = min(f.cap.get(k, math.inf) - used_k[(f.id, k)], f.total_cap - used_j[f.id])
                a = min(rem, room)
                ml = f.min_lot.get(k, 0.0)
                if used_k[(f.id, k)] < 1e-9 and a < ml:
                    continue
                if a <= 1e-9:
                    continue
                alloc[(dp.id, f.id, k)] += a
                used_k[(f.id, k)] += a
                used_j[f.id] += a
                rem -= a
            unshipped[(dp.id, k)] = rem
    return _consolidate(inst, alloc, unshipped)


def _consolidate(inst, alloc, unshipped) -> Plan:
    p = inst.params
    big = max(inst.vehicles, key=lambda v: v.payload_kg)
    days_left = {v.id: v.vehicle_days for v in inst.vehicles}
    trips = []
    for dp in inst.depots:
        per_fac = defaultdict(dict)
        for (o, j, k), q in alloc.items():
            if o == dp.id and q > 1e-9:
                per_fac[j][k] = q
        residual = {}
        for j, streams in per_fac.items():
            # direct full truckloads: fill big trucks container by container
            items = []
            for k, q in streams.items():
                c = inst.container_of(k)
                n = math.ceil(q / c.net_kg - 1e-9)
                for t in range(n):
                    content = min(c.net_kg, q - t * c.net_kg)
                    items.append((k, content, c))
            items.sort(key=lambda it: -(it[1] + it[2].tare_kg))
            # full loads with the largest vehicle that still has vehicle-days left
            while items:
                v = _largest_available(inst, dp.id, (j,), days_left)
                if v is None:
                    for k, content, _c in items:
                        unshipped[(dp.id, k)] = unshipped.get((dp.id, k), 0.0) + content
                    items = []
                    break
                loads = _pack(items, v)
                if len(loads) <= 1:
                    break
                trips.append(_trip(dp.id, v.id, (j,), {0: loads[0]}))
                days_left[v.id] -= trip_cost_model(inst, dp.id, v, (j,)).time.days
                items = [it for ld in loads[1:] for it in ld]
            residual[j] = items
        # merge residuals with a nearest-neighbour sweep
        open_j = [j for j, ld in residual.items() if ld]
        while open_j:
            cur = min(open_j, key=lambda j: inst.d(dp.id, j))
            route, cargo = [cur], list(residual[cur])
            open_j.remove(cur)
            while len(route) < p.max_stops and open_j:
                nxt = min(open_j, key=lambda j: inst.d(route[-1], j))
                cand = _largest_available(inst, dp.id, tuple(route + [nxt]), days_left)
                if cand is None:
                    break
                extra = residual[nxt]
                mass = sum(c + cc.tare_kg for _k, c, cc in cargo + extra)
                slots = sum(cc.slots for _k, _c, cc in cargo + extra)
                if mass > cand.payload_kg or slots > cand.slots:
                    break
                route.append(nxt)
                cargo += extra
                open_j.remove(nxt)
            per_stop = {i: residual[j] for i, j in enumerate(route)}
            veh = _smallest_vehicle(inst, dp.id, tuple(route), per_stop, days_left)
            if veh is None:      # no vehicle left: the cargo stays at the depot
                for ld in per_stop.values():
                    for k, content, _c in ld:
                        unshipped[(dp.id, k)] = unshipped.get((dp.id, k), 0.0) + content
                continue
            days_left[veh] -= trip_cost_model(inst, dp.id, inst.V[veh], tuple(route)).time.days
            trips.append(_trip(dp.id, veh, tuple(route), per_stop))
    return Plan(trips, unshipped, {"solver": "greedy"})


def _pack(items, v):
    loads, cur, mass, slots = [], [], 0.0, 0.0
    for it in items:
        m = it[1] + it[2].tare_kg
        if mass + m > v.payload_kg or slots + it[2].slots > v.slots:
            loads.append(cur)
            cur, mass, slots = [], 0.0, 0.0
        cur.append(it); mass += m; slots += it[2].slots
    loads.append(cur)
    return loads


def _largest_available(inst, depot, stops, days_left):
    best = None
    for v in sorted(inst.vehicles, key=lambda v: -v.payload_kg):
        m = trip_cost_model(inst, depot, v, stops)
        if m.feasible_range and days_left[v.id] >= m.time.days:
            return v
    return best


def _smallest_vehicle(inst, depot, stops, per_stop, days_left=None):
    mass = sum(c + cc.tare_kg for ld in per_stop.values() for _k, c, cc in ld)
    slots = sum(cc.slots for ld in per_stop.values() for _k, _c, cc in ld)
    best, best_cost = None, math.inf
    for v in inst.vehicles:
        if v.payload_kg < mass or v.slots < slots:
            continue
        m = trip_cost_model(inst, depot, v, stops)
        if not m.feasible_range:
            continue
        if days_left is not None and days_left[v.id] < m.time.days:
            continue
        if m.fixed < best_cost:
            best, best_cost = v.id, m.fixed
    return best


def _trip(depot, vid, stops, per_stop) -> Trip:
    deliv, cnt = defaultdict(float), defaultdict(int)
    for i, ld in per_stop.items():
        for k, content, _c in ld:
            deliv[(i, k)] += content
            cnt[(i, k)] += 1
    return Trip(depot, vid, stops, dict(deliv), dict(cnt))
