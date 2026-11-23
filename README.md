# BRLO: Battery Reverse Logistics Optimization

An open-source, three-level decision framework for the reverse logistics of end-of-life (EOL) electric-vehicle and portable batteries. It extends and corrects my bachelor thesis (UC3M, 2024), whose prototype ran on Microsoft Dynamics 365 Business Central.

| Level | Question | Method |
|---|---|---|
| **Strategic** (years) | Which depots should operate, and should we build recycling capacity in Spain instead of shipping to France/Germany? When? | Two-stage stochastic, multi-period MILP over the thesis' pessimistic, neutral and optimistic scenarios; reports VSS and EVPI |
| **Long-haul** (weeks/months) | Which facility receives each waste stream, with which vehicles and on which multi-stop trips? | Exact set-partitioning MILP (HiGHS), two-phase scheme with a **certified gap**, and an ALNS matheuristic |
| **First-mile** (days) | How should collectors route through hundreds of workshops and dealers every day? | Attention model + POMO reinforcement learning (PyTorch) vs. Clarke-Wright and local search |

Long-haul model features: heterogeneous fleet (rigid 12 t and 26 t, 40 t articulated, 19 t battery-electric), six battery chemistries, split deliveries, homologated containers (tare and floor slots), gate fees that can be negative, contracted capacities and minimum lots, **load-dependent fuel and CO₂** (exactly linear in the formulation), and **EU Regulation 561/2006 driving and rest times**.

Every optimiser is audited by the same independent evaluator (`brlo.longhaul.solution.evaluate`), which also prices the original 2024 prototype (`brlo.longhaul.legacy`): it reported €729 for a plan that actually costs about €16.5k and overloads its trucks.

## Results (case study and benchmarks)

| | Result |
|---|---|
| Legacy prototype (2024) | reported €729; true cost €16,511; overloaded trucks (infeasible) |
| Corrected greedy | €15,978 (27.8% above optimum) |
| Two-phase MILP | €11,538 in 9 s, certified gap 0.81%, true gap 0.06% |
| Full MILP | optimum €11,533 proven in 541 s |
| Scalability (< 4,000 columns) | two-phase 3.3% mean gap to the lower bound, ALNS 8.8%, greedy 35.6% |
| Scalability (≈ 8,100 columns) | phase 1 does not converge; ALNS is best (21.8% vs 36.6% greedy) |
| Strategic (2026-2035) | VSS €4.0 M (6.5%), EVPI €7.7 M (12.4%); first Spanish Li plant in 2032-2034 |

The full paper draft is in `paper/main.pdf`, and the interactive map of the case-study plan is in `docs/case_study_map.html`.

## Roadmap

- [ ] Train POMO on a GPU (n = 20/50/100, uniform and clustered) and add `results/e6_firstmile_neural.csv`
- [ ] Seed phase 2 with greedy/ALNS routes so the two-phase scheme survives large instances
- [ ] Stronger ALNS (learned operator selection, e.g. a bandit or an RL policy)
- [ ] Road distances with a local OSRM server (Spain + France + Germany extract)
- [ ] Multi-stage strategic model and battery state-of-health (second life vs recycling)
- [ ] Real gate fees and capacities if a take-back scheme shares them
- [ ] Publish the repository and the map (GitHub Pages)

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                 # core: numpy, scipy (HiGHS), pandas, matplotlib
pip install -e ".[neural]"              # PyTorch for the first-mile policy
pip install -e ".[app]"                 # Streamlit app
pytest -q                               # includes the PyTorch tests when torch is installed
```

No commercial software or licence is needed (HiGHS is bundled with SciPy ≥ 1.9).

## Quick start

```python
from brlo.data.instances import iberia_base
from brlo.longhaul.milp import solve_two_phase
from brlo.longhaul.solution import evaluate
from brlo.viz.maps import plan_map_html

inst = iberia_base()                       # one month at the Madrid depot, 71 t
res = solve_two_phase(inst, time_limit=120)
print(res.objective, res.bound, res.gap)   # upper bound, lower bound, certified gap
print(evaluate(inst, res.plan).as_dict())
open("plan.html", "w").write(plan_map_html(inst, res.plan))
```

```bash
streamlit run app/streamlit_app.py
python -m brlo.firstmile.train --n 20 --epochs 2 --steps 20 --layers 3      # CPU smoke test
python -m brlo.firstmile.train --n 50 --dist clustered --epochs 200 --steps 500   # paper setting (GPU)
python -m brlo.firstmile.train --eval runs/pomo_n50_clustered/best.pt --n 50 --dist clustered
```

## Reproducing the paper

```bash
python experiments/run_longhaul.py e1 e2 e4 e3   # long-haul (~1.5 h on 2 cores)
python experiments/run_strategic.py               # strategic (~10 min)
python experiments/run_firstmile_classical.py     # first-mile baselines
# neural results: train as above, then save results/e6_firstmile_neural.csv with columns
#   dist,n,mean_len,ms_per_inst
python experiments/make_paper_assets.py
cd paper && latexmk -pdf main.tex
```

## Layout

```
src/brlo/
  data/        model.py (dataclasses), instances.py (case study + generator)
  geo/         distance.py (haversine x circuity, OSRM client)
  longhaul/    costs.py (trip cost, EU 561/2006), routes.py (columns), milp.py (MILP, two-phase),
               alns.py (matheuristic), greedy.py, legacy.py (2024 prototype), solution.py (evaluator)
  strategic/   network.py (two-stage stochastic design, VSS/EVPI)
  firstmile/   instances.py, heuristics.py (NN, Clarke-Wright, 2-opt/relocate/swap), neural.py, train.py
  viz/         maps.py (Leaflet HTML)
experiments/   scripts that produce results/ and the paper assets
paper/         LaTeX source
app/           Streamlit app
```

## Data disclaimer

Locations are city-level coordinates of real facilities named in the thesis. ACCUREC's capacities are public. **Gate fees, contracted capacities, stocks and growth scenarios are synthetic** but plausible, because the companies did not disclose them. They are all in `src/brlo/data/instances.py` and `src/brlo/strategic/network.py`, ready to be replaced with real data.

## License

MIT
# route_optimization_thesis
