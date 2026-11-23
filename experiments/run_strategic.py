"""Strategic experiments (E5): stochastic network design and value of the stochastic solution.

    python experiments/run_strategic.py
"""
import json
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brlo.strategic.network import K, default_network, solve_network, stochastic_metrics  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def destination_profile(nd, sol, s):
    """Tonnes of each chemistry treated in Spain vs abroad per year in scenario s."""
    rows = []
    for t, y in enumerate(nd.years):
        for k, kk in enumerate(K):
            es = ab = 0.0
            for (d, j, kk2, tt, ss), v in sol.flows["g"].items():
                if tt == t and ss == s and kk2 == k:
                    if nd.plants[j].id.startswith(("SNAM", "ACC")):
                        ab += v
                    else:
                        es += v
            un = sum(v for (i, k2, tt, ss), v in sol.flows["u"].items() if tt == t and ss == s and k2 == k)
            rows.append({"year": y, "scenario": nd.scenarios[s], "chem": kk, "spain_t": es, "abroad_t": ab, "unserved_t": un})
    return rows


def main():
    nd = default_network()
    m = stochastic_metrics(nd, time_limit=300)
    summary = {k: m[k] for k in ("RP", "EV", "EEV", "WS", "VSS", "EVPI")}
    summary["VSS_pct_of_RP"] = 100 * m["VSS"] / m["RP"]
    summary["EVPI_pct_of_RP"] = 100 * m["EVPI"] / m["RP"]
    designs = {"RP": (m["rp_solution"].depots_open, m["rp_solution"].plants_open),
               "EV": (m["ev_solution"].depots_open, m["ev_solution"].plants_open),
               **{f"WS_{k}": v for k, v in m["ws_designs"].items()}}
    summary["designs"] = designs
    summary["rp_scenario_cost"] = m["rp_solution"].scenario_cost
    summary["eev_scenario_cost"] = m["eev_solution"].scenario_cost
    summary["ws_values"] = m["ws_values"]
    with open(os.path.join(OUT, "e5_strategic.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "designs"}, indent=1, default=float))
    print(json.dumps(designs, indent=1))
    rows = []
    for s in range(len(nd.scenarios)):
        rows += destination_profile(nd, m["rp_solution"], s)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "e5_flows_rp.csv"), index=False)

    # Sensitivity: first year with a Spanish Li plant vs P(optimistic) and plant cost multiplier
    sens = []
    for p_opt in (0.1, 0.25, 0.4):
        for mult in (0.75, 1.0, 1.5):
            nd2 = default_network()
            nd2.prob = np.array([(1 - p_opt) / 3, 2 * (1 - p_opt) / 3, p_opt])
            nd2.plants = [replace(p, fixed_eur=p.fixed_eur * mult) for p in nd2.plants]
            sol = solve_network(nd2, time_limit=300)
            first = min([min(v) for v in sol.plants_open.values() if v], default=None)
            sens.append({"p_optimistic": p_opt, "plant_cost_mult": mult, "objective": sol.objective,
                         "first_plant_year": first, "plants": json.dumps(sol.plants_open)})
            print(sens[-1], flush=True)
    pd.DataFrame(sens).to_csv(os.path.join(OUT, "e5_sensitivity.csv"), index=False)


if __name__ == "__main__":
    main()
