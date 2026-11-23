"""Faithful re-implementation of the 2024 Business Central prototype
(``ArbolEstructura.Page.al``) so that it can be audited with the same
evaluator as the new methods.

Behaviour reproduced on purpose (including its defects):

* each waste stream is planned independently (no consolidation across streams);
* "Option 1" sorts facilities by unit price, "Option 2" by *ascending monthly
  capacity* (the thesis text describes Option 2 as the shortest route, but the
  code never uses distance to order or choose facilities);
* facilities are filled greedily in that order up to their capacity, visited in
  the same order, ignoring minimum lots and facility-wide capacity;
* the reported distance is depot -> s1 -> ... -> sm, without the return leg; the
  previous-stop variable is not reset between options, so Option 2's first leg
  starts from Option 1's last facility;
* vehicle choice filters trucks whose *gross vehicle weight* (MMA) is >= the
  quantity and takes the last one by code; otherwise it walks the fleet once
  in decreasing MMA, so any remainder beyond the sum of MMAs is dropped;
* every truck is charged the whole chain distance at a single consumption
  value; no driver, toll, or rest-time costs.

:func:`legacy_plan` returns both the cost the prototype would have *reported*
and a :class:`~brlo.longhaul.solution.Plan` reflecting what would physically
happen, which the evaluator then prices and checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..data.model import LongHaulInstance
from .solution import Plan, Trip


@dataclass
class LegacyResult:
    option: int
    plan: Plan
    reported_cost: float
    reported_distance: float


def legacy_plan(inst: LongHaulInstance, option: int = 1) -> LegacyResult:
    p = inst.params
    fleet = [v for v in inst.vehicles if v.energy == "diesel"]   # enum only had Gasolina/Diesel
    trips, reported, rep_dist = [], 0.0, 0.0
    shipped = {}
    for dp in inst.depots:
        for k, qty in dp.stock.items():
            if qty <= 0:
                continue
            prev_stop = ""      # CodProvAnter: local to each waste line, but NOT reset between options (bug kept)
            for opt in (1, 2):
                facs = [f for f in inst.facilities if k in f.fees]
                if opt == 1:
                    facs.sort(key=lambda f: f.fees[k])
                else:
                    facs.sort(key=lambda f: f.cap.get(k, math.inf))
                rem = qty
                chain, amounts, total_d = [], [], 0.0
                for f in facs:
                    if rem <= 0:
                        break
                    a = min(rem, f.cap.get(k, math.inf))
                    rem -= a
                    src = prev_stop or dp.id
                    total_d += inst.d(src, f.id)
                    prev_stop = f.id
                    chain.append(f.id)
                    amounts.append(a)
                if opt != option:
                    continue
                fee_cost = sum(inst.F[j].fees[k] * a for j, a in zip(chain, amounts))
                # --- vehicle selection (InsertarFlota) ---
                cand = [v for v in fleet if v.gvw_kg >= qty]
                if cand:
                    trucks = [(sorted(cand, key=lambda v: v.id)[-1], qty)]
                else:
                    trucks, r = [], qty
                    for v in sorted(fleet, key=lambda v: -v.gvw_kg):
                        if r <= 0:
                            break
                        trucks.append((v, min(r, v.gvw_kg)))
                        r -= min(r, v.gvw_kg)
                fuel = sum(p.diesel_eur_l * total_d * v.e_full / 100.0 for v, _ in trucks)
                reported += fee_cost + fuel
                rep_dist += total_d * len(trucks)
                # --- physical plan: each truck runs the chain carrying its share
                c = inst.container_of(k)
                left = list(amounts)
                for v, carried in trucks:
                    deliv, cnt = {}, {}
                    cap_left = carried
                    for i, _j in enumerate(chain):
                        q = min(cap_left, left[i])
                        if q > 1e-9:
                            deliv[(i, k)] = q
                            cnt[(i, k)] = math.ceil(q / c.net_kg - 1e-9)
                            left[i] -= q
                            cap_left -= q
                    if deliv:
                        trips.append(Trip(dp.id, v.id, tuple(chain), deliv, cnt))
                        for (_i, kk), q in deliv.items():
                            shipped[(dp.id, kk)] = shipped.get((dp.id, kk), 0.0) + q
    unshipped = {(dp.id, k): max(0.0, s - shipped.get((dp.id, k), 0.0)) for dp in inst.depots for k, s in dp.stock.items()}
    return LegacyResult(option, Plan(trips, unshipped, {"solver": f"legacy-opt{option}"}), reported, rep_dist)
