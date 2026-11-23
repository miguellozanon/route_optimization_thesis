"""Exact set-partitioning MILP for long-haul dispatch (solved with HiGHS via
``scipy.optimize.milp``).

Sets: depots O, waste streams K, facilities J, vehicle types V, columns R
(trips, see :mod:`brlo.longhaul.routes`); for r in R, stops i = 0..m_r-1 with
facility j(r,i).

Variables
  y_r in Z+        number of trips performed with column r
  x_rik >= 0       kg of stream k delivered at stop i of column r (all its trips)
  n_rik in Z+      containers of stream k delivered at stop i of column r
  z_jk in {0,1}    facility j is used for stream k (only if a minimum lot exists)
  u_ok >= 0        kg of stream k left unshipped at depot o
  w_rk in {0,1}    stream k is carried by column r (only if segregation rules exist)

Objective  min sum_r F_r y_r + sum_rik (f_jk + B_ri) x_rik + sum_rik T_k B_r n_rik + pi sum_ok u_ok
  F_r : load-independent trip cost, B_ri = sum_{l<=i} beta_rl : marginal energy
  (+CO2) cost of carrying 1 kg until stop i, B_r : same for the whole trip
  (tare travels on every leg), f_jk gate fee, pi unshipped penalty.

Constraints
  (supply)    sum_{r in R_o} sum_i x_rik + u_ok = s_ok
  (capacity)  sum x_rik [j(r,i)=j] <= Q_jk z_jk ;  >= m_jk z_jk ;  sum_k ... <= Q_j
  (packing)   x_rik <= c_k n_rik
  (payload)   sum_ik x_rik + sum_ik T_k n_rik <= P_v y_r
  (floor)     sum_ik S_k n_rik <= S_v y_r
  (fleet)     sum_{r: v(r)=v} d_r y_r <= H_v      (vehicle-days)
  (segregate) x_rik <= P_v ub_r w_rk ; w_rk + w_rk' <= 1 for incompatible (k,k')

The model aggregates the y_r identical trips of a column. A solution is turned
into individual trips by :func:`disaggregate`, which packs containers into
the y_r vehicles (first-fit decreasing on mass with a floor-slot check); the
evaluator then audits the result.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ..data.model import LongHaulInstance
from .routes import Column, enumerate_columns
from .solution import Plan, Trip


class LinearModel:
    """Tiny sparse model builder on top of scipy.optimize.milp."""

    def __init__(self) -> None:
        self.c, self.lb, self.ub, self.integ = [], [], [], []
        self.rows, self.cols, self.vals = [], [], []
        self.rlb, self.rub = [], []

    def var(self, cost=0.0, lb=0.0, ub=np.inf, integer=False) -> int:
        self.c.append(cost); self.lb.append(lb); self.ub.append(ub); self.integ.append(1 if integer else 0)
        return len(self.c) - 1

    def constr(self, coefs: dict, lb=-np.inf, ub=np.inf) -> int:
        r = len(self.rlb)
        for j, a in coefs.items():
            if a != 0:
                self.rows.append(r); self.cols.append(j); self.vals.append(a)
        self.rlb.append(lb); self.rub.append(ub)
        return r

    def solve(self, time_limit=60.0, gap=1e-4, verbose=False):
        n = len(self.c)
        A = coo_matrix((self.vals, (self.rows, self.cols)), shape=(len(self.rlb), n)).tocsr()
        cons = [LinearConstraint(A, np.array(self.rlb), np.array(self.rub))] if self.rlb else []
        return milp(np.array(self.c), integrality=np.array(self.integ), bounds=Bounds(np.array(self.lb), np.array(self.ub)),
                    constraints=cons, options={"time_limit": time_limit, "mip_rel_gap": gap, "disp": verbose})


@dataclass
class MILPResult:
    plan: Plan
    objective: float
    bound: float
    gap: float
    status: str
    seconds: float
    n_columns: int
    n_vars: int
    n_cons: int
    n_int: int
    stats: dict = field(default_factory=dict)


def build_and_solve(inst: LongHaulInstance, columns: list[Column] | None = None, time_limit: float = 120.0,
                    gap: float = 1e-4, y_upper: dict | None = None, verbose: bool = False,
                    max_stops: int | None = None, all_orders: bool = False, strengthen: bool = True,
                    relax_containers: bool = False) -> MILPResult:
    """Solve the dispatch MILP over ``columns`` (all columns if None).

    ``y_upper`` optionally caps y_r per column index (used by the ALNS
    matheuristic to evaluate a restricted route set)."""
    t0 = time.perf_counter()
    if columns is None:
        columns = enumerate_columns(inst, max_stops=max_stops, all_orders=all_orders)
    p = inst.params
    M = LinearModel()
    incompatible = [tuple(x) for x in inst.incompatible]

    supply = defaultdict(dict)      # (o,k) -> coefs
    capk = defaultdict(dict)        # (j,k)
    capj = defaultdict(dict)        # j
    fleet = defaultdict(dict)       # v
    yv, xv, nv = {}, {}, {}

    for col in columns:
        v = inst.V[col.vehicle]
        stock = inst.D[col.depot].stock
        ub = col.max_trips if y_upper is None else min(col.max_trips, y_upper.get(col.idx, 0))
        if ub <= 0:
            continue
        y = M.var(cost=col.model.fixed, ub=ub, integer=True)
        yv[col.idx] = y
        fleet[v.id][y] = col.model.time.days
        Bcum = np.cumsum(col.model.beta)          # B_ri = sum_{l<=i} beta_l
        Btot = float(sum(col.model.beta))
        payload = {y: -v.payload_kg}
        floor = {y: -v.slots}
        streams = set()
        stop_vub = {}
        per_ctype = {}
        for i, jid in enumerate(col.stops):
            fac = inst.F[jid]
            for k, fee in fac.fees.items():
                if stock.get(k, 0) <= 0:
                    continue
                c = inst.container_of(k)
                xub = min(stock[k], fac.cap.get(k, np.inf), fac.total_cap, v.payload_kg * ub)
                x = M.var(cost=fee + float(Bcum[i]), ub=xub)
                n = M.var(cost=c.tare_kg * Btot, ub=np.floor(v.slots * ub / c.slots), integer=not relax_containers)
                xv[(col.idx, i, k)] = x
                nv[(col.idx, i, k)] = n
                supply[(col.depot, k)][x] = 1.0
                capk[(jid, k)][x] = 1.0
                capj[jid][x] = 1.0
                M.constr({x: 1.0, n: -c.net_kg}, ub=0.0)
                if strengthen:   # variable upper bound: one trip carries at most U_rik of k to stop i
                    U = min(stock[k], fac.cap.get(k, np.inf), fac.total_cap, v.payload_kg - c.tare_kg)
                    M.constr({x: 1.0, y: -U}, ub=0.0)
                    stop_vub.setdefault(i, {})[x] = 1.0
                payload[x] = 1.0
                payload[n] = c.tare_kg
                floor[n] = c.slots
                per_ctype.setdefault(c.id, {})[n] = 1.0
                streams.add(k)
        M.constr(payload, ub=0.0)
        M.constr(floor, ub=0.0)
        for cid, coefs in per_ctype.items():      # per-truck integrality of floor positions
            cc = inst.C[cid]
            per_truck = int(np.floor(v.slots / cc.slots + 1e-9))
            if per_truck * cc.slots < v.slots - 1e-9:
                M.constr({**coefs, y: -per_truck}, ub=0.0)
        if strengthen:
            tot = {}
            for i, coefs in stop_vub.items():
                Uj = min(inst.F[col.stops[i]].total_cap, v.payload_kg)
                M.constr({**coefs, y: -Uj}, ub=0.0)
                tot.update(coefs)
            avail = sum(stock[k] for k in streams)
            if tot and avail < v.payload_kg:
                M.constr({**tot, y: -avail}, ub=0.0)
        if incompatible:
            wv = {}
            for k in streams:
                wv[k] = M.var(ub=1, integer=True)
                for (cidx, i, kk), x in xv.items():
                    if cidx == col.idx and kk == k:
                        M.constr({x: 1.0, wv[k]: -v.payload_kg * ub}, ub=0.0)
            for a, b in incompatible:
                if a in wv and b in wv:
                    M.constr({wv[a]: 1.0, wv[b]: 1.0}, ub=1.0)

    uvars = {}
    for dp in inst.depots:
        for k, s in dp.stock.items():
            if s <= 0:
                continue
            u = M.var(cost=p.unshipped_penalty_eur_kg, ub=s)
            uvars[(dp.id, k)] = u
            coefs = dict(supply[(dp.id, k)])
            coefs[u] = 1.0
            M.constr(coefs, lb=s, ub=s)
    for f in inst.facilities:
        for k in f.fees:
            if (f.id, k) not in capk:
                continue
            avail = sum(dp.stock.get(k, 0.0) for dp in inst.depots)
            coefs = dict(capk[(f.id, k)])
            ml = f.min_lot.get(k, 0.0)
            capk_val = f.cap.get(k, np.inf)
            if ml > 0:
                z = M.var(ub=1, integer=True)
                M.constr({**coefs, z: -min(capk_val, f.total_cap, avail)}, ub=0.0)
                M.constr({**coefs, z: -ml}, lb=0.0)
            elif np.isfinite(capk_val):
                M.constr(coefs, ub=capk_val)
        if f.id in capj and np.isfinite(f.total_cap):
            M.constr(capj[f.id], ub=f.total_cap)
    for vid, coefs in fleet.items():
        M.constr(coefs, ub=inst.V[vid].vehicle_days)

    res = M.solve(time_limit=time_limit, gap=gap, verbose=verbose)
    secs = time.perf_counter() - t0
    status = {0: "optimal", 1: "time/iteration limit", 2: "infeasible", 3: "unbounded", 4: "other"}.get(res.status, str(res.status))
    if res.x is None:
        return MILPResult(Plan(meta={"status": status}), np.inf, getattr(res, "mip_dual_bound", np.nan), np.inf,
                          status, secs, len(columns), len(M.c), len(M.rlb), sum(M.integ))
    X = res.x
    colmap = {c.idx: c for c in columns}
    trips = []
    agg = []
    for cidx, y in yv.items():
        ny = int(round(X[y]))
        if ny <= 0:
            continue
        col = colmap[cidx]
        deliv = {(i, k): X[x] for (ci, i, k), x in xv.items() if ci == cidx and X[x] > 1e-6}
        conts = {(i, k): int(np.ceil(X[nv[(ci, i, k)]] - 1e-6)) for (ci, i, k) in xv if ci == cidx
                 and np.ceil(X[nv[(ci, i, k)]] - 1e-6) > 0}
        agg.append((col.idx, ny))
        trips.extend(disaggregate(inst, col, ny, deliv, conts))
    unshipped = {key: X[u] for key, u in uvars.items() if X[u] > 1e-6}
    for dp in inst.depots:
        for k, s in dp.stock.items():
            unshipped.setdefault((dp.id, k), 0.0)
    plan = Plan(trips=trips, unshipped=unshipped, meta={"solver": "milp", "status": status})
    bound = getattr(res, "mip_dual_bound", None)
    bound = float(res.fun) if bound is None else float(bound)
    gapv = getattr(res, "mip_gap", None)
    gapv = 0.0 if gapv is None else float(gapv)
    return MILPResult(plan, float(res.fun), bound, gapv, status, secs, len(columns), len(M.c), len(M.rlb),
                      int(sum(M.integ)), stats={"aggregated_columns": len(agg), "y": agg})


def disaggregate(inst: LongHaulInstance, col: Column, ny: int, deliv: dict, conts: dict) -> list[Trip]:
    """Split the aggregate decision of a column into ``ny`` concrete trips.

    First try first-fit decreasing (containers are items with mass = tare +
    content and floor slots). If any vehicle would be overloaded, solve the
    exact 2-D assignment MILP: a_bg containers of group g = (stop, stream) in
    truck b, q_bg kg of content, with sum_b a_bg = n_g, sum_b q_bg = x_g,
    q_bg <= net_g a_bg and mass/slot limits per truck. Trips that end up empty
    are dropped and stops without deliveries removed (never increases cost)."""
    v = inst.V[col.vehicle]
    groups = [(g, n, deliv.get(g, 0.0)) for g, n in conts.items() if n > 0]
    alloc = _ffd(inst, v, ny, groups)
    if alloc is None:
        alloc = _exact_split(inst, v, ny, groups)
    if alloc is None:      # aggregate not disaggregable: keep FFD result, evaluator will flag it
        alloc = _ffd(inst, v, ny, groups, force=True)
    trips = []
    for cnt, q in alloc:
        if not cnt:
            continue
        used = sorted({i for (i, _k) in cnt})
        remap = {old: new for new, old in enumerate(used)}
        stops = tuple(col.stops[i] for i in used)
        trips.append(Trip(col.depot, col.vehicle, stops,
                          {(remap[i], k): qq for (i, k), qq in q.items() if qq > 1e-9},
                          {(remap[i], k): n for (i, k), n in cnt.items()}))
    return trips


def _ffd(inst, v, ny, groups, force=False):
    items = []
    for (i, k), n, q in groups:
        c = inst.container_of(k)
        per = q / n
        items += [(per + c.tare_kg, c.slots, (i, k), per)] * n
    best = None
    for key in (lambda t: -t[0], lambda t: -t[1], lambda t: -(t[0] / v.payload_kg + t[1] / v.slots)):
        load = [0.0] * ny; slots = [0.0] * ny
        cnt = [defaultdict(int) for _ in range(ny)]; qq = [defaultdict(float) for _ in range(ny)]
        ok = True
        for mass, sl, g, per in sorted(items, key=key):
            fit = [b for b in range(ny) if load[b] + mass <= v.payload_kg + 1e-6 and slots[b] + sl <= v.slots + 1e-6]
            if fit:
                b = min(fit, key=lambda b: (load[b] / v.payload_kg + slots[b] / v.slots))
            else:
                ok = False
                b = min(range(ny), key=lambda b: load[b])
            load[b] += mass; slots[b] += sl; cnt[b][g] += 1; qq[b][g] += per
        if ok:
            return list(zip(cnt, qq))
        if best is None:
            best = list(zip(cnt, qq))
    return best if force else None


def _exact_split(inst, v, ny, groups):
    M = LinearModel()
    A, Q = {}, {}
    for b in range(ny):
        mass, slot = {}, {}
        for gi, ((i, k), n, q) in enumerate(groups):
            c = inst.container_of(k)
            A[b, gi] = M.var(ub=n, integer=True)
            Q[b, gi] = M.var(ub=q)
            M.constr({Q[b, gi]: 1.0, A[b, gi]: -c.net_kg}, ub=0)
            mass[Q[b, gi]] = 1.0; mass[A[b, gi]] = c.tare_kg; slot[A[b, gi]] = c.slots
        M.constr(mass, ub=v.payload_kg); M.constr(slot, ub=v.slots)
    for gi, (g, n, q) in enumerate(groups):
        M.constr({A[b, gi]: 1.0 for b in range(ny)}, lb=n, ub=n)
        M.constr({Q[b, gi]: 1.0 for b in range(ny)}, lb=q, ub=q)
    res = M.solve(time_limit=10)
    if res.x is None:
        return None
    out = []
    for b in range(ny):
        cnt, qq = defaultdict(int), defaultdict(float)
        for gi, (g, n, q) in enumerate(groups):
            a = int(round(res.x[A[b, gi]]))
            if a > 0:
                cnt[g] += a; qq[g] += res.x[Q[b, gi]]
        out.append((cnt, qq))
    return out


def lp_bound(inst: LongHaulInstance, columns: list[Column]) -> float:
    """Linear-programming relaxation value: a valid lower bound in any case."""
    orig = LinearModel.solve

    def solve_lp(self, time_limit=60.0, gap=1e-4, verbose=False):
        self.integ = [0] * len(self.integ)
        return orig(self, time_limit, gap, verbose)
    LinearModel.solve = solve_lp
    try:
        r = build_and_solve(inst, columns, time_limit=300)
    finally:
        LinearModel.solve = orig
    return r.objective


def solve_two_phase(inst: LongHaulInstance, time_limit: float = 120.0, gap: float = 1e-4,
                    max_stops: int | None = None, all_orders: bool = False, slack_trips: int = 1,
                    neighbourhood: bool = True) -> MILPResult:
    """Certified two-phase scheme.

    Phase 1 solves the model with container counts relaxed to continuous
    values. Every feasible solution of the full model is feasible for this
    relaxation, so its optimal value (or its dual bound) is a valid LOWER
    bound. Phase 2 re-solves the full model (integer containers) restricted
    to the columns used in phase 1, allowing ``slack_trips`` extra trips per
    column, which yields a feasible plan, i.e. an UPPER bound. The reported
    gap (UB - LB) / UB certifies solution quality."""
    t0 = time.perf_counter()
    cols = enumerate_columns(inst, max_stops=max_stops, all_orders=all_orders)
    r1 = build_and_solve(inst, cols, time_limit=time_limit / 2, gap=gap, relax_containers=True)
    lb = r1.bound
    if r1.status != "optimal" or not np.isfinite(lb):
        lb = max(lb if np.isfinite(lb) else -np.inf, lp_bound(inst, cols))
    yv = _phase1_y(inst, cols, r1)
    y_upper = {c: n + slack_trips for c, n in yv.items()}
    if neighbourhood:   # same stop set with other vehicles, and every subset of used stop sets
        used_sets = {(cols[c].depot, frozenset(cols[c].stops)) for c in yv}
        for c in cols:
            key = (c.depot, frozenset(c.stops))
            if c.idx in y_upper:
                continue
            if any(c.depot == dp and key[1] <= S for dp, S in used_sets):
                y_upper[c.idx] = slack_trips
    r2 = build_and_solve(inst, cols, time_limit=time_limit / 2, gap=gap, y_upper=y_upper)
    ub = r2.objective
    secs = time.perf_counter() - t0
    r2.plan.meta.update({"solver": "two-phase", "lb": lb, "ub": ub})
    return MILPResult(r2.plan, ub, lb, (ub - lb) / max(1e-9, abs(ub)), "certified", secs,
                      len(cols), r1.n_vars, r1.n_cons, r1.n_int, stats={"phase1_status": r1.status,
                                                                       "phase2_status": r2.status,
                                                                       "restricted_columns": len(y_upper)})


def _phase1_y(inst, cols, r1) -> dict:
    """Map the trips of a phase-1 plan back to column indices (y_r)."""
    key = {(c.depot, c.vehicle, c.stops): c.idx for c in cols}
    out = defaultdict(int)
    for agg in r1.stats.get("y", []):
        out[agg[0]] += agg[1]
    if out:
        return dict(out)
    for t in r1.plan.trips:
        k = (t.depot, t.vehicle, t.stops)
        if k in key:
            out[key[k]] += 1
    return dict(out)
