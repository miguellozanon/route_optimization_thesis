"""Core data model shared by every optimisation level.

Units used throughout the package (unless stated otherwise):
mass in kg, distance in km, time in hours, money in EUR, emissions in kg CO2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class WasteType:
    """A battery waste stream (chemistry family) that must be routed to a
    compatible treatment facility.

    ``container`` is the id of the homologated packaging used for this stream
    (ADR packaging instructions are chemistry-specific, so each stream has one
    default container type).
    """

    id: str
    name: str
    container: str
    adr_class: str = "9"        # informative only (UN3480/UN3090 -> class 9, Pb-acid -> 8)


@dataclass(frozen=True)
class Container:
    """Homologated packaging. ``net_kg`` is the maximum *content* mass, ``tare_kg``
    the empty mass and ``slots`` the floor positions (EUR-pallet equivalents)."""

    id: str
    name: str
    net_kg: float
    tare_kg: float
    slots: float


@dataclass(frozen=True)
class VehicleType:
    """Vehicle class.

    The original Business Central prototype used the gross vehicle weight
    (MMA) as if it were the cargo capacity; here ``payload_kg`` is the
    legal *cargo* capacity and ``gvw_kg`` is kept only to reproduce that
    behaviour in :mod:`brlo.longhaul.legacy`.

    Energy use is linear in the carried mass (``e_empty`` at 0 kg,
    ``e_full`` at ``payload_kg``), expressed in energy units per 100 km
    (litres of diesel or kWh).
    """

    id: str
    name: str
    payload_kg: float
    slots: float
    gvw_kg: float
    energy: str                  # "diesel" | "electric"
    e_empty: float               # units / 100 km at zero load
    e_full: float                # units / 100 km at full payload
    fixed_cost: float = 120.0    # EUR per trip (dispatch, ADR paperwork, cleaning)
    cost_per_km: float = 0.20    # EUR / km (tolls + maintenance + tyres)
    speed_kmh: float = 70.0      # average long-haul speed
    vehicle_days: float = 30.0   # available vehicle-days in the planning period
    daily_range_km: Optional[float] = None   # electric trucks: km per day incl. break charging


@dataclass(frozen=True)
class Site:
    id: str
    name: str
    lat: float
    lon: float
    country: str


@dataclass(frozen=True)
class Depot(Site):
    """Temporary storage centre holding sorted waste ready for long-haul."""

    stock: dict = field(default_factory=dict)   # waste_id -> kg available this period


@dataclass(frozen=True)
class Facility(Site):
    """Treatment (recycling / second-life) facility.

    ``fees``      waste_id -> gate fee EUR/kg (negative = the facility pays,
                  which is typical for cobalt/nickel-rich chemistries).
    ``cap``       waste_id -> contracted capacity in kg for the period.
    ``total_cap`` overall kg the facility accepts from us in the period.
    ``min_lot``   minimum kg per waste type if the facility is used for it.
    """

    fees: dict = field(default_factory=dict)
    cap: dict = field(default_factory=dict)
    total_cap: float = float("inf")
    min_lot: dict = field(default_factory=dict)
    service_h: float = 1.5
    kind: str = "recycling"      # "recycling" | "second-life"


@dataclass
class Params:
    diesel_eur_l: float = 1.45
    electricity_eur_kwh: float = 0.25
    co2_kg_per_l_diesel: float = 2.68     # tank-to-wheel
    co2_kg_per_kwh: float = 0.11          # grid intensity (parameter)
    driver_eur_h: float = 25.0            # loaded labour cost (paid time)
    per_diem_eur: float = 60.0            # per night away from base
    co2_price_eur_t: float = 0.0          # shadow price of CO2 (0 = pure cost)
    unshipped_penalty_eur_kg: float = 3.0 # storage + regulatory risk of leaving stock
    max_stops: int = 3
    # EU Regulation (EC) No 561/2006 (single driver)
    max_continuous_driving_h: float = 4.5
    break_h: float = 0.75
    max_daily_driving_h: float = 9.0
    daily_rest_h: float = 11.0


@dataclass
class LongHaulInstance:
    name: str
    depots: list
    facilities: list
    waste_types: list
    containers: list
    vehicles: list
    params: Params
    dist: dict          # (site_id, site_id) -> km
    incompatible: list = field(default_factory=list)  # [(waste_id, waste_id)] may not share a truck

    # ----- convenience lookups -------------------------------------------
    def __post_init__(self) -> None:
        self.W = {w.id: w for w in self.waste_types}
        self.C = {c.id: c for c in self.containers}
        self.V = {v.id: v for v in self.vehicles}
        self.F = {f.id: f for f in self.facilities}
        self.D = {d.id: d for d in self.depots}

    def d(self, a: str, b: str) -> float:
        return 0.0 if a == b else self.dist[(a, b)]

    def container_of(self, waste_id: str) -> Container:
        return self.C[self.W[waste_id].container]

    def total_stock(self) -> float:
        return sum(sum(dp.stock.values()) for dp in self.depots)
