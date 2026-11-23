"""Two-stage stochastic, multi-period reverse-logistics network design.

Question answered (the core question of the original thesis): *as the
volume of end-of-life batteries grows, which storage depots should operate
and should treatment capacity be built in Spain instead of shipping to France
and Germany - and when?*

Echelons: collection regions i -> storage depots d -> treatment facilities j,
for chemistry groups k in {LI, NI, PB}, years t = 0..T-1, scenarios s with
probability p_s.

First stage (here-and-now, scenario independent)
  a_dt in {0,1}  depot d operates in year t        (a_dt >= a_d,t-1)
  b_jt in {0,1}  candidate plant j operates in year t (b_jt >= b_j,t-1,
                 b_jt = 0 for t < lead time)
Second stage (per scenario)
  f_idkts >= 0   tonnes collected in region i and sent to depot d
  g_djkts >= 0   tonnes shipped from depot d to facility j
  u_ikts  >= 0   tonnes not collected/treated (penalised)

min  sum_t delta^t [ sum_d FD_d a_dt + sum_j FP_j b_jt ]
   + sum_s p_s sum_t delta^t [ sum c1_idk f + sum (c2_dj + fee_jk) g + pi u ]
s.t. sum_d f_idkts + u_ikts = D_ikts                    (demand)
     sum_i f_idkts = sum_j g_djkts                       (depot balance)
     sum_ik f_idkts <= K_d a_dt                          (depot capacity)
     sum_d g_djkts <= Q_jk (b_jt if candidate else 1)    (treatment capacity)
     sum_dk g_djkts <= Q_j (b_jt if candidate else 1)

Per-tonne transport costs are not guessed: the depot -> facility rate comes
from the long-haul trip model (EU 561/2006 times, tolls, load-dependent fuel)
of :mod:`brlo.longhaul.costs`, and the first-mile rate uses the classical
continuum approximation of collection tour length
L ~ 2 r N / C + k sqrt(N A) (Beardwood, Halton & Hammersley, 1959; Daganzo, 1984).

Reported metrics (Birge & Louveaux, 2011): RP (recourse problem), WS
(wait-and-see), EEV (expected result of the expected-value solution),
EVPI = RP - WS and VSS = EEV - RP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..data.instances import VEHICLES
from ..data.model import Depot, Facility, LongHaulInstance, Params
from ..geo.distance import haversine_km
from ..longhaul.costs import trip_cost_model
from ..longhaul.milp import LinearModel

K = ("LI", "NI", "PB")

# Peninsular Spain, NUTS-2 regions: (id, capital, lat, lon, approx. population share, area km2)
REGIONS = [
    ("AND", "Sevilla", 37.39, -5.98, 0.183, 87_600), ("CAT", "Barcelona", 41.39, 2.17, 0.170, 32_100),
    ("MAD", "Madrid", 40.42, -3.70, 0.149, 8_000), ("VAL", "Valencia", 39.47, -0.38, 0.112, 23_300),
    ("GAL", "Santiago", 42.88, -8.54, 0.057, 29_600), ("CYL", "Valladolid", 41.65, -4.72, 0.051, 94_200),
    ("PVA", "Bilbao", 43.26, -2.93, 0.047, 7_200), ("CLM", "Toledo", 39.86, -4.02, 0.044, 79_400),
    ("MUR", "Murcia", 37.99, -1.13, 0.034, 11_300), ("ARA", "Zaragoza", 41.65, -0.88, 0.029, 47_700),
    ("EXT", "Merida", 38.92, -6.34, 0.022, 41_600), ("AST", "Oviedo", 43.36, -5.85, 0.021, 10_600),
    ("NAV", "Pamplona", 42.81, -1.65, 0.014, 10_400), ("CAN", "Santander", 43.46, -3.80, 0.012, 5_300),
    ("RIO", "Logrono", 42.47, -2.45, 0.007, 5_000),
]


@dataclass
class DepotCand:
    id: str
    lat: float
    lon: float
    cap_t: float          # t/year
    fixed_eur: float      # EUR/year when operating
    existing: bool = False


@dataclass
class PlantCand:
    id: str
    name: str
    lat: float
    lon: float
    cap_t: dict           # k -> t/year
    total_t: float
    fee: dict             # k -> EUR/t (negative = revenue)
    fixed_eur: float = 0.0   # EUR/year if candidate (annualised capex + fixed opex)
    candidate: bool = False
    lead: int = 0         # earliest operating year index


@dataclass
class NetworkData:
    years: list
    scenarios: list           # names
    prob: np.ndarray
    demand: np.ndarray        # [i, k, t, s] tonnes
    depots: list
    plants: list
    c1: np.ndarray            # [i, d] EUR/t region -> depot (line-haul + local collection)
    c2: np.ndarray            # [d, j] EUR/t depot -> plant
    discount: float = 0.04
    penalty: float = 3000.0   # EUR/t uncollected (regulatory non-compliance)


def _ftl_rate_eur_t(lat1, lon1, lat2, lon2, circuity=1.25, load_factor=0.85) -> float:
    """EUR per tonne of a direct full-truckload trip (tractor-trailer), using
    the same trip-cost model as the operational level."""
    if haversine_km(lat1, lon1, lat2, lon2) < 1.0:
        return 0.0
    a = Depot("A", "A", lat1, lon1, "ES")
    b = Facility("B", "B", lat2, lon2, "XX", fees={})
    km = circuity * haversine_km(lat1, lon1, lat2, lon2)
    inst = LongHaulInstance("tmp", [a], [b], [], [], list(VEHICLES), Params(),
                            {("A", "B"): km, ("B", "A"): km})
    v = next(x for x in VEHICLES if x.id == "T40")
    m = trip_cost_model(inst, "A", v, ("B",))
    q = load_factor * v.payload_kg
    return (m.fixed + m.beta[0] * q) / (q / 1000.0)


def first_mile_rate_eur_t(area_km2: float, n_points: int, kg_per_pickup: float, cap_kg: float = 3500.0,
                          eur_km: float = 1.1, eur_stop: float = 12.0, circuity: float = 1.3) -> float:
    """Continuum approximation of local collection cost per tonne inside a region.
    Average radial distance r ~ 0.38 sqrt(A) (centroid depot, compact region)."""
    r = 0.38 * math.sqrt(area_km2)
    stops_per_tour = max(1.0, cap_kg / kg_per_pickup)
    length_per_pickup = circuity * (2 * r / stops_per_tour + 0.7124 * math.sqrt(area_km2 / n_points))
    return (eur_km * length_per_pickup + eur_stop) / (kg_per_pickup / 1000.0)


def default_network(T: int = 10, start_year: int = 2026) -> NetworkData:
    years = list(range(start_year, start_year + T))
    scen = ["pessimistic", "neutral", "optimistic"]
    prob = np.array([0.25, 0.5, 0.25])
    # Operator's collected tonnes in year 0 and annual growth per scenario (ILLUSTRATIVE,
    # calibrated to the qualitative shape of the thesis' three EV-demand scenarios).
    base = {"LI": 3_000.0, "NI": 900.0, "PB": 8_000.0}
    growth = {"LI": (0.06, 0.17, 0.30), "NI": (-0.03, -0.02, -0.01), "PB": (0.0, 0.0, 0.0)}
    nI = len(REGIONS)
    D = np.zeros((nI, len(K), T, len(scen)))
    for i, reg in enumerate(REGIONS):
        for k, kk in enumerate(K):
            for t in range(T):
                for s in range(len(scen)):
                    D[i, k, t, s] = reg[4] * base[kk] * (1 + growth[kk][s]) ** t

    depots = [DepotCand("MAD", 40.33, -3.71, 30_000, 350_000, existing=True),
              DepotCand("BCN", 41.33, 2.10, 20_000, 300_000), DepotCand("VLC", 39.45, -0.39, 15_000, 250_000),
              DepotCand("SVQ", 37.36, -5.94, 15_000, 250_000), DepotCand("BIO", 43.26, -2.92, 12_000, 220_000),
              DepotCand("ZAZ", 41.65, -0.88, 12_000, 220_000)]
    plants = [
        PlantCand("SNAM_VIV", "SNAM Viviez", 44.556, 2.218, {"LI": 3_000, "NI": 600}, 3_500, {"LI": 350, "NI": 650}),
        PlantCand("SNAM_SQF", "SNAM St-Quentin-F.", 45.64, 5.11, {"LI": 2_500}, 2_500, {"LI": 330}),
        PlantCand("ACC_KRE", "ACCUREC Krefeld", 51.334, 6.563, {"LI": 5_000, "NI": 800}, 5_500, {"LI": 250, "NI": 550}),
        PlantCand("ACC_MUL", "ACCUREC Muelheim", 51.427, 6.883, {"NI": 400}, 400, {"NI": 500}),
        PlantCand("MINURB", "Mineria Urbana", 40.948, -4.118, {"LI": 1_200}, 1_200, {"LI": 420}),
        PlantCand("RECYP", "Recypilas", 43.30, -2.95, {"LI": 600, "NI": 500, "PB": 12_000}, 12_500,
                  {"LI": 650, "NI": 700, "PB": -120}),
        PlantCand("NEW_LEON", "New Li plant Leon (planned)", 42.60, -5.57, {"LI": 8_000}, 8_000, {"LI": -50},
                  fixed_eur=5_200_000, candidate=True, lead=2),
        PlantCand("NEW_ZAZ", "New Li plant Zaragoza", 41.65, -0.88, {"LI": 10_000}, 10_000, {"LI": -50},
                  fixed_eur=6_300_000, candidate=True, lead=3),
        PlantCand("NEW_SVQ", "New Li plant Sevilla", 37.39, -5.98, {"LI": 6_000}, 6_000, {"LI": -50},
                  fixed_eur=4_400_000, candidate=True, lead=3),
    ]
    c1 = np.zeros((nI, len(depots)))
    for i, reg in enumerate(REGIONS):
        local = first_mile_rate_eur_t(reg[5], n_points=max(20, int(400 * reg[4] / 0.05)), kg_per_pickup=250)
        for d, dp in enumerate(depots):
            c1[i, d] = local + _ftl_rate_eur_t(reg[2], reg[3], dp.lat, dp.lon) * 1.15   # partial loads
    c2 = np.array([[_ftl_rate_eur_t(dp.lat, dp.lon, pl.lat, pl.lon) for pl in plants] for dp in depots])
    return NetworkData(years, scen, prob, D, depots, plants, c1, c2)


@dataclass
class NetworkSolution:
    objective: float
    first_stage_cost: float
    expected_second_stage: float
    depots_open: dict          # depot -> list of years open
    plants_open: dict          # plant -> list of years open
    scenario_cost: dict
    flows: dict = field(default_factory=dict)
    status: str = ""
    seconds: float = 0.0


def solve_network(nd: NetworkData, scenarios: list | None = None, prob: np.ndarray | None = None,
                  demand: np.ndarray | None = None, fix_first_stage: dict | None = None,
                  time_limit: float = 120.0, gap: float = 1e-5) -> NetworkSolution:
    """Extensive form of the two-stage model. ``fix_first_stage`` = {"a": {(d,t):0/1}, "b": {(j,t):0/1}}
    fixes the design (used to compute EEV)."""
    import time
    t0 = time.perf_counter()
    D = nd.demand if demand is None else demand
    S = list(range(D.shape[3])) if scenarios is None else scenarios
    pr = (nd.prob if prob is None else prob)
    nI, nK, T = D.shape[0], D.shape[1], D.shape[2]
    disc = [(1 + nd.discount) ** -t for t in range(T)]
    M = LinearModel()
    a, b = {}, {}
    for d, dp in enumerate(nd.depots):
        for t in range(T):
            fixed_val = None
            if dp.existing:
                fixed_val = 1
            if fix_first_stage is not None:
                fixed_val = fix_first_stage["a"][(d, t)]
            lo = hi = fixed_val if fixed_val is not None else None
            a[d, t] = M.var(cost=disc[t] * dp.fixed_eur, lb=lo if lo is not None else 0, ub=hi if hi is not None else 1, integer=True)
            if t > 0:
                M.constr({a[d, t]: 1, a[d, t - 1]: -1}, lb=0)
    for j, pl in enumerate(nd.plants):
        if not pl.candidate:
            continue
        for t in range(T):
            ub = 0 if t < pl.lead else 1
            lb = 0
            if fix_first_stage is not None:
                lb = ub = fix_first_stage["b"][(j, t)]
            b[j, t] = M.var(cost=disc[t] * pl.fixed_eur, lb=lb, ub=ub, integer=True)
            if t > 0:
                M.constr({b[j, t]: 1, b[j, t - 1]: -1}, lb=0)
    f, g, u = {}, {}, {}
    acc = [[k for k, kk in enumerate(K) if kk in pl.cap_t] for pl in nd.plants]
    for si, s in enumerate(S):
        ps = pr[si] if len(pr) == len(S) else pr[s]
        for t in range(T):
            w = ps * disc[t]
            for d in range(len(nd.depots)):
                depcap = {}
                for i in range(nI):
                    for k in range(nK):
                        f[i, d, k, t, s] = M.var(cost=w * nd.c1[i, d])
                        depcap[f[i, d, k, t, s]] = 1.0
                depcap[a[d, t]] = -nd.depots[d].cap_t
                M.constr(depcap, ub=0)
                for j in range(len(nd.plants)):
                    for k in acc[j]:
                        g[d, j, k, t, s] = M.var(cost=w * (nd.c2[d, j] + nd.plants[j].fee[K[k]]))
                for k in range(nK):
                    bal = {f[i, d, k, t, s]: 1.0 for i in range(nI)}
                    for j in range(len(nd.plants)):
                        if k in acc[j]:
                            bal[g[d, j, k, t, s]] = -1.0
                    M.constr(bal, lb=0, ub=0)
            for i in range(nI):
                for k in range(nK):
                    u[i, k, t, s] = M.var(cost=w * nd.penalty)
                    coefs = {f[i, d, k, t, s]: 1.0 for d in range(len(nd.depots))}
                    coefs[u[i, k, t, s]] = 1.0
                    M.constr(coefs, lb=D[i, k, t, s], ub=D[i, k, t, s])
            for j, pl in enumerate(nd.plants):
                tot = {}
                for k in acc[j]:
                    coefs = {g[d, j, k, t, s]: 1.0 for d in range(len(nd.depots))}
                    tot.update(coefs)
                    if pl.candidate:
                        coefs[b[j, t]] = -pl.cap_t[K[k]]
                        M.constr(coefs, ub=0)
                    else:
                        M.constr(coefs, ub=pl.cap_t[K[k]])
                if pl.candidate:
                    tot[b[j, t]] = -pl.total_t
                    M.constr(tot, ub=0)
                else:
                    M.constr(tot, ub=pl.total_t)
    res = M.solve(time_limit=time_limit, gap=gap)
    X = res.x
    first = sum(M.c[v] * X[v] for v in list(a.values()) + list(b.values()))
    sc_cost = {}
    for si, s in enumerate(S):
        ps = pr[si] if len(pr) == len(S) else pr[s]
        tot = 0.0
        for key, v in list(f.items()) + list(g.items()):
            if key[-1] == s:
                tot += M.c[v] * X[v]
        for key, v in u.items():
            if key[-1] == s:
                tot += M.c[v] * X[v]
        sc_cost[nd.scenarios[s]] = tot / ps if ps > 0 else 0.0
    dep_open = {nd.depots[d].id: [nd.years[t] for t in range(T) if X[a[d, t]] > 0.5] for d in range(len(nd.depots))}
    pl_open = {nd.plants[j].id: [nd.years[t] for t in range(T) if X[b[j, t]] > 0.5]
               for j in range(len(nd.plants)) if nd.plants[j].candidate}
    flows = {"g": {k: X[v] for k, v in g.items() if X[v] > 1e-6},
             "u": {k: X[v] for k, v in u.items() if X[v] > 1e-6},
             "a": {k: int(round(X[v])) for k, v in a.items()},
             "b": {k: int(round(X[v])) for k, v in b.items()}}
    return NetworkSolution(float(res.fun), first, float(res.fun) - first, dep_open, pl_open, sc_cost, flows,
                           "optimal" if res.status == 0 else str(res.status), time.perf_counter() - t0)


def stochastic_metrics(nd: NetworkData, time_limit: float = 300.0) -> dict:
    """RP, EV, EEV, WS, VSS and EVPI for the network design problem."""
    nS = nd.demand.shape[3]
    rp = solve_network(nd, time_limit=time_limit)
    mean_D = np.tensordot(nd.demand, nd.prob, axes=([3], [0]))[..., None]
    ev = solve_network(nd, demand=mean_D, prob=np.array([1.0]), scenarios=[0], time_limit=time_limit)
    full_b = {(j, t): 0 for j, pl in enumerate(nd.plants) if pl.candidate for t in range(len(nd.years))}
    full_b.update(ev.flows["b"])
    eev = solve_network(nd, fix_first_stage={"a": ev.flows["a"], "b": full_b}, time_limit=time_limit)
    ws_vals, ws_designs = [], {}
    for s in range(nS):
        Ds = nd.demand[..., s:s + 1]
        sol = solve_network(nd, demand=Ds, prob=np.array([1.0]), scenarios=[0], time_limit=time_limit)
        ws_vals.append(sol.objective)
        ws_designs[nd.scenarios[s]] = (sol.depots_open, sol.plants_open)
    ws = float(np.dot(nd.prob, ws_vals))
    return {"RP": rp.objective, "EV": ev.objective, "EEV": eev.objective, "WS": ws,
            "VSS": eev.objective - rp.objective, "EVPI": rp.objective - ws,
            "rp_solution": rp, "ev_solution": ev, "eev_solution": eev, "ws_designs": ws_designs,
            "ws_values": dict(zip(nd.scenarios, ws_vals))}
