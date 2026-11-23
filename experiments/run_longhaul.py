"""Long-haul experiments (E1 case study, E2 ablation, E3 scalability, E4 sensitivity).

    python experiments/run_longhaul.py e1 e2 e4      # ~10 min
    python experiments/run_longhaul.py e3            # ~45 min on 2 cores
Results are written to results/*.csv and results/*.json.
"""
import json
import os
import sys
import time
from dataclasses import replace

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from brlo.data.instances import iberia_base, random_instance          # noqa: E402
from brlo.data.model import Params                                     # noqa: E402
from brlo.longhaul.alns import ALNS                                    # noqa: E402
from brlo.longhaul.greedy import greedy_plan                           # noqa: E402
from brlo.longhaul.legacy import legacy_plan                           # noqa: E402
from brlo.longhaul.milp import build_and_solve, lp_bound, solve_two_phase  # noqa: E402
from brlo.longhaul.routes import enumerate_columns                     # noqa: E402
from brlo.longhaul.solution import evaluate                            # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def plan_rows(inst, plan):
    rows = []
    for t in plan.trips:
        rows.append({"depot": t.depot, "vehicle": t.vehicle, "stops": "-".join(t.stops),
                     "cargo_kg": round(t.cargo_kg()),
                     "streams": ",".join(sorted({k for (_i, k) in t.deliveries}))})
    return rows


def e1():
    inst = iberia_base()
    rows, plans = [], {}
    for opt in (1, 2):
        L = legacy_plan(inst, opt)
        e = evaluate(inst, L.plan)
        rows.append({"method": f"Legacy BC prototype, option {opt}", "reported_cost": L.reported_cost,
                     **e.as_dict(), "seconds": 0.0, "lb": None})
        plans[f"legacy{opt}"] = (plan_rows(inst, L.plan), e.violations)
    t = time.perf_counter(); g = greedy_plan(inst); tg = time.perf_counter() - t
    eg = evaluate(inst, g)
    rows.append({"method": "Greedy (corrected)", **eg.as_dict(), "seconds": tg, "lb": None})
    a = ALNS(inst, seed=0, eval_time=3).run(g, iterations=400, time_limit=120)
    rows.append({"method": "ALNS from greedy (120 s)", **evaluate(inst, a.plan).as_dict(), "seconds": a.seconds, "lb": None})
    tp = solve_two_phase(inst, time_limit=120)
    etp = evaluate(inst, tp.plan)
    rows.append({"method": "Two-phase MILP (certified)", **etp.as_dict(), "seconds": tp.seconds, "lb": tp.bound})
    plans["two_phase"] = (plan_rows(inst, tp.plan), etp.violations)
    full = build_and_solve(inst, time_limit=600, gap=1e-4)
    ef = evaluate(inst, full.plan)
    rows.append({"method": "Full MILP (600 s)", **ef.as_dict(), "seconds": full.seconds, "lb": full.bound})
    best_lb = max(tp.bound, full.bound)
    df = pd.DataFrame(rows)
    df["gap_to_best_lb_pct"] = 100 * (df["cost"] - best_lb) / df["cost"]
    df.to_csv(os.path.join(OUT, "e1_case_study.csv"), index=False)
    best = min([(etp.cost, "two_phase", tp.plan), (ef.cost, "full", full.plan)], key=lambda x: x[0])
    with open(os.path.join(OUT, "e1_plans.json"), "w") as fh:
        json.dump({**plans, "best": (plan_rows(inst, best[2]), []), "best_source": best[1], "best_lb": best_lb}, fh, indent=1)
    print(df[["method", "cost", "transport", "gate_fees", "co2_kg", "n_trips", "feasible", "n_violations", "seconds"]].round(1))


def e2():
    """Ablation: effect of the valid inequalities and of container integrality."""
    rows = []
    for name, inst in [("iberia_base", iberia_base()), ("rand_f12_s0", random_instance(12, 1, seed=0))]:
        cols = enumerate_columns(inst)
        for strengthen in (False, True):
            import brlo.longhaul.milp as mm
            t = time.perf_counter()
            lp = _lp(inst, cols, strengthen)
            r = build_and_solve(inst, cols, time_limit=120, strengthen=strengthen)
            rc = build_and_solve(inst, cols, time_limit=120, strengthen=strengthen, relax_containers=True)
            rows.append({"instance": name, "strengthen": strengthen, "lp_root": lp,
                         "full_ub": r.objective, "full_lb": r.bound, "full_s": r.seconds,
                         "relaxed_containers_opt": rc.objective, "relaxed_status": rc.status, "relaxed_s": rc.seconds})
            print(rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "e2_ablation.csv"), index=False)


def _lp(inst, cols, strengthen):
    import brlo.longhaul.milp as mm
    orig = mm.LinearModel.solve
    def lp(self, time_limit=60.0, gap=1e-4, verbose=False):
        self.integ = [0] * len(self.integ)
        return orig(self, time_limit, gap, verbose)
    mm.LinearModel.solve = lp
    try:
        return mm.build_and_solve(inst, cols, time_limit=300, strengthen=strengthen).objective
    finally:
        mm.LinearModel.solve = orig


def e3(seeds=(0, 1, 2)):
    path = os.path.join(OUT, "e3_scalability.csv")
    grid = [(8, 1), (12, 1), (16, 1), (12, 2), (20, 2)]
    done = set()
    if os.path.exists(path):
        old = pd.read_csv(path)
        done = {(r.n_fac, r.n_dep, r.seed, r.method) for r in old.itertuples()}
    for nf, nd in grid:
        for s in seeds:
            inst = random_instance(nf, nd, seed=s)
            ncols = len(enumerate_columns(inst))
            rows = []
            t = time.perf_counter(); g = greedy_plan(inst); tg = time.perf_counter() - t
            rows.append(("greedy", evaluate(inst, g), tg, None))
            a = ALNS(inst, seed=s, eval_time=3).run(g, iterations=10_000, time_limit=60)
            rows.append(("alns", evaluate(inst, a.plan), a.seconds, None))
            tp = solve_two_phase(inst, time_limit=120)
            rows.append(("two_phase", evaluate(inst, tp.plan), tp.seconds, tp.bound))
            a2 = ALNS(inst, seed=s, eval_time=3).run(tp.plan, iterations=10_000, time_limit=45)
            rows.append(("two_phase+alns", evaluate(inst, a2.plan), tp.seconds + a2.seconds, tp.bound))
            out = []
            for m, e, secs, lb in rows:
                if (nf, nd, s, m) in done:
                    continue
                out.append({"n_fac": nf, "n_dep": nd, "seed": s, "columns": ncols, "method": m, "cost": e.cost,
                            "feasible": e.feasible, "unshipped_kg": e.unshipped_kg, "seconds": secs,
                            "lb": tp.bound})
            df = pd.DataFrame(out)
            df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(df.round(1).to_string(index=False), flush=True)


def e4():
    rows = []
    for co2 in (0, 100, 250, 500):
        inst = iberia_base(params=Params(co2_price_eur_t=co2))
        r = solve_two_phase(inst, time_limit=90)
        e = evaluate(inst, r.plan)
        # report cost WITHOUT the shadow price to obtain the cost/CO2 trade-off
        inst0 = iberia_base()
        e0 = evaluate(inst0, r.plan)
        veh = {}
        for t in r.plan.trips:
            veh[t.vehicle] = veh.get(t.vehicle, 0) + 1
        spain = sum(q for t in r.plan.trips for (i, k), q in t.deliveries.items() if inst.F[t.stops[i]].country == "ES")
        rows.append({"study": "co2_price", "value": co2, "cost_eur": e0.cost, "co2_kg": e0.co2_kg,
                     "gap_pct": 100 * r.gap, "share_spain": spain / max(1, e0.shipped_kg), **{f"trips_{k}": v for k, v in veh.items()}})
        print(rows[-1])
    for scale in (0.5, 1.0, 2.0, 3.0):
        inst = iberia_base(stock_scale=scale)
        r = solve_two_phase(inst, time_limit=120)
        e = evaluate(inst, r.plan)
        spain = sum(q for t in r.plan.trips for (i, k), q in t.deliveries.items() if inst.F[t.stops[i]].country == "ES")
        rows.append({"study": "stock_scale", "value": scale, "cost_eur": e.cost, "co2_kg": e.co2_kg,
                     "eur_per_t": 1000 * e.cost / max(1, inst.total_stock()), "gap_pct": 100 * r.gap,
                     "unshipped_kg": e.unshipped_kg, "share_spain": spain / max(1, e.shipped_kg)})
        print(rows[-1])
    for circ in (1.15, 1.25, 1.35):
        inst = iberia_base(circuity=circ)
        r = solve_two_phase(inst, time_limit=90)
        e = evaluate(inst, r.plan)
        rows.append({"study": "circuity", "value": circ, "cost_eur": e.cost, "co2_kg": e.co2_kg, "gap_pct": 100 * r.gap})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "e4_sensitivity.csv"), index=False)


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["e1", "e2", "e4", "e3"]:
        print(f"===== {arg} =====", flush=True)
        globals()[arg]()
