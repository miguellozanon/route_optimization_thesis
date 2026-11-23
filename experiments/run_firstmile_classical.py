"""First-mile classical baselines (E6a). The neural results (E6b) are produced with
    python -m brlo.firstmile.train ...   and   python -m brlo.firstmile.train --eval ...
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brlo.firstmile import instances as gen                                   # noqa: E402
from brlo.firstmile.heuristics import check, dist_matrix, solve_classical, total_length  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def main(size=100):
    rows = []
    for dist in ("uniform", "clustered"):
        for n in (20, 50, 100):
            b = (gen.clustered if dist == "clustered" else gen.uniform)(size, n, seed=999)
            for m in ("nn", "cw", "cw+ls"):
                L, t0 = [], time.perf_counter()
                for i in range(size):
                    r = solve_classical(b.coords[i], b.demand[i], b.capacity, m)
                    check(r, b.demand[i], b.capacity)
                    L.append(total_length(r, dist_matrix(b.coords[i])))
                rows.append({"dist": dist, "n": n, "method": m, "mean_len": float(np.mean(L)),
                             "ms_per_inst": 1000 * (time.perf_counter() - t0) / size,
                             "km_120km_region": float(np.mean(L)) * 120 * 1.3})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "e6_firstmile_classical.csv"), index=False)


if __name__ == "__main__":
    main()
