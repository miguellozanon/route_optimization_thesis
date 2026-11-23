"""Instances.

``iberia_base()`` is the case study derived from the original thesis: one
temporary storage depot near Madrid shipping sorted battery waste to the two
recyclers Ecopilas actually used in 2024 (SNAM in France, ACCUREC in Germany)
and to the Spanish facilities identified in the thesis (Section 3.4).

Data provenance (important for honest reporting):
* Locations: city-level coordinates of the facilities named in the thesis
  (approximate, +/- a few km).
* ACCUREC capacities: public figures (Krefeld 60 kt/y, Muelheim 4 kt/y); only a
  contracted fraction is assumed available to one client.
* Gate fees, contracted capacities, minimum lots and stocks: SYNTHETIC but
  plausible values (no company disclosed them). Negative fees model
  chemistries with positive recovered-metal value (NMC/NCA, Pb).
  Every number lives here so that it can be replaced by real data.

``random_instance()`` generates scalable instances for computational
experiments (multi-depot, many facilities) with the same structure.
"""

from __future__ import annotations

import random
from dataclasses import replace

from ..geo.distance import road_matrix_haversine
from .model import (Container, Depot, Facility, LongHaulInstance, Params,
                    VehicleType, WasteType)

WASTE_TYPES = [
    WasteType("NMC", "Li-ion NMC/NCA (traction)", "EVBOX", "9"),
    WasteType("LFP", "Li-ion LFP (traction)", "EVBOX", "9"),
    WasteType("LIP", "Li-ion portable", "ADR600", "9"),
    WasteType("NIMH", "Ni-MH (hybrid packs + portable)", "ADR600", "9"),
    WasteType("NICD", "Ni-Cd (portable/industrial)", "ADR600", "8"),
    WasteType("PB", "Lead-acid (automotive SLI)", "PBBOX", "8"),
]

CONTAINERS = [
    Container("EVBOX", "Fire-retardant steel box for EV modules", net_kg=600, tare_kg=220, slots=2.0),
    Container("ADR600", "UN-homologated 600 L plastic box", net_kg=450, tare_kg=40, slots=1.0),
    Container("PBBOX", "Acid-proof pallet box", net_kg=900, tare_kg=45, slots=1.0),
]

VEHICLES = [
    VehicleType("R12", "Rigid 12 t", payload_kg=5_500, slots=15, gvw_kg=12_000, energy="diesel",
                e_empty=19.0, e_full=24.0, fixed_cost=100, cost_per_km=0.16, vehicle_days=20),
    VehicleType("R26", "Rigid 26 t", payload_kg=14_500, slots=18, gvw_kg=26_000, energy="diesel",
                e_empty=24.0, e_full=31.0, fixed_cost=120, cost_per_km=0.20, vehicle_days=20),
    VehicleType("T40", "Tractor + semitrailer 40 t", payload_kg=24_000, slots=33, gvw_kg=40_000,
                energy="diesel", e_empty=23.0, e_full=33.0, fixed_cost=150, cost_per_km=0.24,
                vehicle_days=20),
    VehicleType("E19", "Battery-electric rigid 19 t", payload_kg=8_000, slots=16, gvw_kg=19_000,
                energy="electric", e_empty=95.0, e_full=125.0, fixed_cost=110, cost_per_km=0.14,
                vehicle_days=20, daily_range_km=450),
]

# (id, name, lat, lon, country, kind)
SITES = {
    "MAD": ("Ecopilas storage (Madrid area, assumed)", 40.33, -3.71, "ES"),
    "SNAM_VIV": ("SNAM Viviez", 44.556, 2.218, "FR"),
    "SNAM_SQF": ("SNAM Saint-Quentin-Fallavier", 45.640, 5.110, "FR"),
    "ACC_KRE": ("ACCUREC Krefeld", 51.334, 6.563, "DE"),
    "ACC_MUL": ("ACCUREC Muelheim an der Ruhr", 51.427, 6.883, "DE"),
    "MINURB": ("Mineria Urbana (Segovia)", 40.948, -4.118, "ES"),
    "RECYP": ("Recypilas (Bizkaia)", 43.300, -2.950, "ES"),
    "TERA": ("Tera Batteries (Alicante)", 38.345, -0.481, "ES"),
    "BEEPL": ("BeePlanet (Navarra)", 42.760, -1.640, "ES"),
    "REFACT": ("Renault Refactory (Sevilla)", 37.390, -5.980, "ES"),
}


def _facilities() -> list:
    t = 1000.0
    return [
        Facility("SNAM_VIV", *SITES["SNAM_VIV"],
                 fees={"NMC": -0.25, "LFP": 0.85, "LIP": 0.55, "NIMH": 0.35, "NICD": 0.95},
                 cap={"NMC": 25 * t, "LFP": 15 * t, "LIP": 10 * t, "NIMH": 10 * t, "NICD": 8 * t},
                 total_cap=40 * t, min_lot={k: 1000 for k in ("NMC", "LFP", "LIP", "NIMH", "NICD")}),
        Facility("SNAM_SQF", *SITES["SNAM_SQF"],
                 fees={"NMC": -0.30, "LFP": 0.80, "LIP": 0.50},
                 cap={"NMC": 20 * t, "LFP": 12 * t, "LIP": 8 * t},
                 total_cap=30 * t, min_lot={k: 1000 for k in ("NMC", "LFP", "LIP")}),
        Facility("ACC_KRE", *SITES["ACC_KRE"],
                 fees={"NMC": -0.45, "LFP": 0.65, "LIP": 0.45, "NIMH": 0.25, "NICD": 0.80},
                 cap={"NMC": 60 * t, "LFP": 40 * t, "LIP": 30 * t, "NIMH": 20 * t, "NICD": 15 * t},
                 total_cap=120 * t, min_lot={k: 2000 for k in ("NMC", "LFP", "LIP", "NIMH", "NICD")}),
        Facility("ACC_MUL", *SITES["ACC_MUL"],
                 fees={"NIMH": 0.22, "NICD": 0.75, "LIP": 0.50},
                 cap={"NIMH": 8 * t, "NICD": 6 * t, "LIP": 5 * t},
                 total_cap=12 * t, min_lot={k: 1000 for k in ("NIMH", "NICD", "LIP")}),
        Facility("MINURB", *SITES["MINURB"],
                 fees={"NMC": -0.05, "LFP": 1.10, "LIP": 0.70},
                 cap={"NMC": 6 * t, "LFP": 4 * t, "LIP": 4 * t},
                 total_cap=8 * t, min_lot={k: 500 for k in ("NMC", "LFP", "LIP")}),
        Facility("RECYP", *SITES["RECYP"],
                 fees={"LIP": 0.75, "NIMH": 0.45, "NICD": 1.10, "PB": -0.12},
                 cap={"LIP": 6 * t, "NIMH": 5 * t, "NICD": 4 * t, "PB": 60 * t},
                 total_cap=60 * t, min_lot={k: 500 for k in ("LIP", "NIMH", "NICD", "PB")}),
        Facility("TERA", *SITES["TERA"], kind="second-life",
                 fees={"NMC": 0.10, "LFP": 0.20},
                 cap={"NMC": 3 * t, "LFP": 3 * t}, total_cap=4 * t,
                 min_lot={"NMC": 500, "LFP": 500}),
        Facility("BEEPL", *SITES["BEEPL"], kind="second-life",
                 fees={"NMC": 0.05, "LFP": 0.15},
                 cap={"NMC": 4 * t, "LFP": 4 * t}, total_cap=6 * t,
                 min_lot={"NMC": 1000, "LFP": 1000}),
        Facility("REFACT", *SITES["REFACT"], kind="second-life",
                 fees={"NMC": 0.00, "LFP": 0.30},
                 cap={"NMC": 3 * t, "LFP": 2 * t}, total_cap=4 * t,
                 min_lot={"NMC": 500, "LFP": 500}),
    ]


def iberia_base(stock_scale: float = 1.0, circuity: float = 1.25, params: Params | None = None) -> LongHaulInstance:
    """One planning period (a month) of the Madrid storage depot."""
    stock = {"NMC": 18_000, "LFP": 9_000, "LIP": 7_500, "NIMH": 4_000, "NICD": 2_500, "PB": 30_000}
    stock = {k: v * stock_scale for k, v in stock.items()}
    depot = Depot("MAD", *SITES["MAD"], stock=stock)
    facs = _facilities()
    dist = road_matrix_haversine([depot, *facs], circuity)
    return LongHaulInstance("iberia_base", [depot], facs, list(WASTE_TYPES), list(CONTAINERS),
                            list(VEHICLES), params or Params(), dist)


# ---------------------------------------------------------------------------
# Random instances for scalability experiments
# ---------------------------------------------------------------------------
_EU_BOX = (36.5, 53.0, -8.5, 15.0)   # lat_min, lat_max, lon_min, lon_max
_DEPOT_CITIES = [("MAD", 40.33, -3.71), ("BCN", 41.33, 2.10), ("VLC", 39.45, -0.39),
                 ("SVQ", 37.36, -5.94), ("BIO", 43.26, -2.92), ("ZAZ", 41.65, -0.88)]


def random_instance(n_facilities: int = 15, n_depots: int = 1, seed: int = 0,
                    stock_scale: float = 1.0, max_stops: int = 3, circuity: float = 1.25) -> LongHaulInstance:
    rng = random.Random(seed)
    wids = [w.id for w in WASTE_TYPES]
    base_fee = {"NMC": -0.2, "LFP": 0.8, "LIP": 0.6, "NIMH": 0.35, "NICD": 0.9, "PB": -0.1}
    depots = []
    for i in range(n_depots):
        cid, lat, lon = _DEPOT_CITIES[i % len(_DEPOT_CITIES)]
        st = {w: rng.uniform(0.4, 1.6) * b * stock_scale for w, b in
              {"NMC": 15_000, "LFP": 8_000, "LIP": 6_000, "NIMH": 3_500, "NICD": 2_000, "PB": 25_000}.items()}
        depots.append(Depot(f"{cid}{i}", f"Depot {cid}", lat, lon, "ES", stock=st))
    facs = []
    for j in range(n_facilities):
        lat = rng.uniform(_EU_BOX[0], _EU_BOX[1])
        lon = rng.uniform(_EU_BOX[2], _EU_BOX[3])
        k = rng.randint(2, 5)
        acc = rng.sample(wids, k)
        fees = {w: round(base_fee[w] + rng.uniform(-0.25, 0.35), 3) for w in acc}
        cap = {w: rng.choice([4, 6, 10, 15, 25]) * 1000.0 for w in acc}
        facs.append(Facility(f"F{j:03d}", f"Facility {j}", lat, lon, "EU", fees=fees, cap=cap,
                             total_cap=0.7 * sum(cap.values()),
                             min_lot={w: rng.choice([0, 500, 1000, 2000]) for w in acc}))
    p = Params(max_stops=max_stops)
    dist = road_matrix_haversine([*depots, *facs], circuity)
    vehicles = [replace(v, vehicle_days=20 * n_depots) for v in VEHICLES]
    return LongHaulInstance(f"rand_f{n_facilities}_d{n_depots}_s{seed}", depots, facs,
                            list(WASTE_TYPES), list(CONTAINERS), vehicles, p, dist)
