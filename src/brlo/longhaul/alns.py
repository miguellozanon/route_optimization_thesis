"""ALNS matheuristic for large long-haul instances.

When the number of facilities, depots or stops per trip grows, the full
column pool of :mod:`brlo.longhaul.routes` explodes (O(|O| |V| sum_k C(|J|,k) k!)).
The matheuristic searches over a small *route skeleton* - a multiset of trip
templates (depot, vehicle, ordered stops) - and evaluates each skeleton
exactly with the dispatch MILP restricted to it (which decides loads,
containers, facility usage and how many of the allowed trips to run). This
follows the Adaptive Large Neighbourhood Search of Ropke & Pisinger (2006)
with simulated-annealing acceptance; the MILP-based evaluation makes it a
matheuristic in the sense of Maniezzo, Stuetzle & Voss (2009).

State      : dict template -> max trips allowed
Destroy    : random removal, worst-utilisation removal, facility removal
Repair     : add direct template, insert facility into template (cheapest
             insertion), merge two templates, change vehicle type, add a trip
Acceptance : simulated annealing, T0 = 1% of initial cost, geometric cooling
Weights    : roulette wheel, scores (33, 9, 13) per segment of 50 iterations
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..data.model import LongHaulInstance
from .costs import trip_cost_model
from .milp import build_and_solve
from .routes import Column, best_order
from .solution import Plan

Template = tuple   # (depot, vehicle, stops)


@dataclass
class ALNSResult:
    plan: Plan
    cost: float
    history: list = field(default_factory=list)
    seconds: float = 0.0
    evaluations: int = 0
    op_stats: dict = field(default_factory=dict)


class ALNS:
    def __init__(self, inst: LongHaulInstance, seed: int = 0, eval_time: float = 5.0):
        self.inst = inst
        self.rng = random.Random(seed)
        self.eval_time = eval_time
        self.cache: dict = {}
        self.col_cache: dict = {}
        self.evals = 0
        p = inst.params
        self.L = p.max_stops
        self.useful = {dp.id: [f.id for f in inst.facilities if any(dp.stock.get(w, 0) > 0 for w in f.fees)]
                       for dp in inst.depots}
        self.destroy = [self.d_random, self.d_worst, self.d_facility]
        self.repair = [self.r_direct, self.r_insert, self.r_merge, self.r_vehicle, self.r_more]

    # ------------------------------------------------------------------ eval
    def _column(self, tpl: Template, idx: int) -> Column | None:
        if tpl not in self.col_cache:
            dp, vid, stops = tpl
            v = self.inst.V[vid]
            m = trip_cost_model(self.inst, dp, v, stops)
            if not m.feasible_range:
                self.col_cache[tpl] = None
            else:
                self.col_cache[tpl] = (m, int(v.vehicle_days // m.time.days))
        ent = self.col_cache[tpl]
        if ent is None or ent[1] <= 0:
            return None
        return Column(idx, tpl[0], tpl[1], tpl[2], ent[0], ent[1])

    def evaluate(self, state: dict):
        key = tuple(sorted(state.items()))
        if key in self.cache:
            return self.cache[key]
        cols, yub = [], {}
        for tpl, n in state.items():
            c = self._column(tpl, len(cols))
            if c is not None and n > 0:
                cols.append(c)
                yub[c.idx] = n
        self.evals += 1
        if not cols:
            pen = self.inst.params.unshipped_penalty_eur_kg * self.inst.total_stock()
            out = (pen, Plan(unshipped={(d.id, k): s for d in self.inst.depots for k, s in d.stock.items()}), {})
        else:
            r = build_and_solve(self.inst, cols, time_limit=self.eval_time, gap=1e-4, y_upper=yub)
            used = Counter()
            for cidx, ny in r.stats.get("y", []):
                used[(cols[cidx].depot, cols[cidx].vehicle, cols[cidx].stops)] = ny
            out = (r.objective, r.plan, dict(used))
        self.cache[key] = out
        return out

    # --------------------------------------------------------------- helpers
    def _norm(self, dp, stops) -> tuple:
        return best_order(self.inst, dp, tuple(dict.fromkeys(stops)))

    def _add(self, state, tpl, n=1):
        state[tpl] = state.get(tpl, 0) + n

    # --------------------------------------------------------------- destroy
    def d_random(self, s, used):
        s = dict(s)
        for _ in range(self.rng.randint(1, max(1, len(s) // 3))):
            if not s:
                break
            t = self.rng.choice(list(s))
            s[t] -= 1
            if s[t] <= 0:
                del s[t]
        return s

    def d_worst(self, s, used):
        s = dict(s)
        unused = [t for t in s if used.get(t, 0) < s[t]]
        if unused:                                   # drop slack first
            for t in unused:
                s[t] = used.get(t, 0)
                if s[t] <= 0:
                    del s[t]
            return s
        if s:
            def per_trip(t):
                m = self._column(t, 0)
                return m.model.fixed / max(1, self.inst.V[t[1]].payload_kg) if m else 1e9
            t = max(s, key=per_trip)
            s[t] -= 1
            if s[t] <= 0:
                del s[t]
        return s

    def d_facility(self, s, used):
        facs = sorted({j for t in s for j in t[2]})
        if not facs:
            return dict(s)
        j = self.rng.choice(facs)
        out = {}
        for (dp, v, stops), n in s.items():
            ns = tuple(x for x in stops if x != j)
            if ns:
                self._add(out, (dp, v, self._norm(dp, ns)), n)
        return out

    # ---------------------------------------------------------------- repair
    def r_direct(self, s):
        s = dict(s)
        dp = self.rng.choice(self.inst.depots).id
        if not self.useful[dp]:
            return s
        j = self.rng.choice(self.useful[dp])
        v = self.rng.choice(self.inst.vehicles).id
        self._add(s, (dp, v, (j,)))
        return s

    def r_insert(self, s):
        s = dict(s)
        cand = [t for t in s if len(t[2]) < self.L]
        if not cand:
            return self.r_direct(s)
        t = self.rng.choice(cand)
        dp, v, stops = t
        options = [j for j in self.useful[dp] if j not in stops]
        if not options:
            return s
        def ins_cost(j):
            best = math.inf
            path = (dp, *stops, dp)
            for i in range(len(path) - 1):
                best = min(best, self.inst.d(path[i], j) + self.inst.d(j, path[i + 1]) - self.inst.d(path[i], path[i + 1]))
            return best
        k = min(3, len(options))
        j = self.rng.choice(sorted(options, key=ins_cost)[:k])
        n = s.pop(t)
        if n > 1:
            s[t] = n - 1
        self._add(s, (dp, v, self._norm(dp, (*stops, j))))
        return s

    def r_merge(self, s):
        s = dict(s)
        by_dp = defaultdict(list)
        for t in s:
            by_dp[t[0]].append(t)
        pairs = [(a, b) for ts in by_dp.values() for a in ts for b in ts
                 if a < b and len(set(a[2]) | set(b[2])) <= self.L]
        if not pairs:
            return self.r_insert(s)
        a, b = self.rng.choice(pairs)
        for t in (a, b):
            s[t] -= 1
            if s[t] <= 0:
                del s[t]
        v = max((a[1], b[1]), key=lambda vid: self.inst.V[vid].payload_kg)
        self._add(s, (a[0], v, self._norm(a[0], (*a[2], *b[2]))))
        return s

    def r_vehicle(self, s):
        s = dict(s)
        if not s:
            return self.r_direct(s)
        t = self.rng.choice(list(s))
        v = self.rng.choice([x.id for x in self.inst.vehicles if x.id != t[1]])
        s[t] -= 1
        if s[t] <= 0:
            del s[t]
        self._add(s, (t[0], v, t[2]))
        return s

    def r_more(self, s):
        s = dict(s)
        if not s:
            return self.r_direct(s)
        self._add(s, self.rng.choice(list(s)))
        return s

    # ------------------------------------------------------------------ main
    def run(self, init: Plan, iterations: int = 300, time_limit: float = 120.0,
            segment: int = 50, cooling: float = 0.995, verbose: bool = False) -> ALNSResult:
        t0 = time.perf_counter()
        state = {}
        for t in init.trips:
            self._add(state, (t.depot, t.vehicle, self._norm(t.depot, t.stops)))
        cur_cost, cur_plan, cur_used = self.evaluate(state)
        best = (cur_cost, cur_plan, state)
        T = 0.01 * abs(cur_cost)
        wd = [1.0] * len(self.destroy)
        wr = [1.0] * len(self.repair)
        sd = [0.0] * len(self.destroy); nd = [0] * len(self.destroy)
        sr = [0.0] * len(self.repair); nr = [0] * len(self.repair)
        hist = [(0, time.perf_counter() - t0, cur_cost, best[0])]
        stats = {f.__name__: [0, 0] for f in self.destroy + self.repair}
        for it in range(1, iterations + 1):
            if time.perf_counter() - t0 > time_limit:
                break
            di = self.rng.choices(range(len(self.destroy)), wd)[0]
            ri = self.rng.choices(range(len(self.repair)), wr)[0]
            cand = self.repair[ri](self.destroy[di](state, cur_used))
            if self.rng.random() < 0.3:
                cand = self.repair[self.rng.randrange(len(self.repair))](cand)
            c_cost, c_plan, c_used = self.evaluate(cand)
            score = 0
            if c_cost < best[0] - 1e-6:
                best = (c_cost, c_plan, cand); score = 33
            if c_cost < cur_cost - 1e-6:
                score = max(score, 9)
            accept = c_cost < cur_cost or self.rng.random() < math.exp(-(c_cost - cur_cost) / max(T, 1e-9))
            if accept:
                if score == 0:
                    score = 13 if c_cost <= cur_cost + 1e-6 else 0
                state, cur_cost, cur_used = cand, c_cost, c_used
                # keep the skeleton tight: shrink allowance to what is used
                state = {t: max(1, n) for t, n in state.items() if c_used.get(t, 0) > 0} or state
            sd[di] += score; nd[di] += 1; sr[ri] += score; nr[ri] += 1
            stats[self.destroy[di].__name__][0] += 1; stats[self.repair[ri].__name__][0] += 1
            if score >= 9:
                stats[self.destroy[di].__name__][1] += 1; stats[self.repair[ri].__name__][1] += 1
            T *= cooling
            if it % segment == 0:
                for w, sc, n in ((wd, sd, nd), (wr, sr, nr)):
                    for i in range(len(w)):
                        if n[i]:
                            w[i] = 0.8 * w[i] + 0.2 * sc[i] / n[i]
                        w[i] = max(w[i], 0.05)
                        sc[i] = 0.0; n[i] = 0
            hist.append((it, time.perf_counter() - t0, cur_cost, best[0]))
            if verbose and it % 25 == 0:
                print(f"it {it:4d}  cur {cur_cost:10.1f}  best {best[0]:10.1f}  T {T:7.2f}  evals {self.evals}")
        plan = best[1]
        plan.meta.update({"solver": "alns"})
        return ALNSResult(plan, best[0], hist, time.perf_counter() - t0, self.evals, stats)
