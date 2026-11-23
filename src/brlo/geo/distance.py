"""Distance providers.

* :func:`haversine_km` - great-circle distance.
* :func:`road_matrix_haversine` - great-circle x circuity factor. Empirical
  road-network circuity in Europe is typically 1.2-1.4 (Ballou et al., 2002);
  1.25 is used as default and exposed as a parameter for sensitivity analysis.
* :class:`OSRMClient` - real road distances from an OSRM server (BSD-2,
  OpenStreetMap data under ODbL). Results are cached on disk so that
  experiments are reproducible offline.

The original prototype mixed Google Distance Matrix (for costs) and Bing Maps
(for display), which produced inconsistent distances; here a single provider
feeds both the optimiser and the map.
"""

from __future__ import annotations

import json
import math
import os
from typing import Iterable, Sequence

R_EARTH_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def road_matrix_haversine(sites: Sequence, circuity: float = 1.25) -> dict:
    out = {}
    for a in sites:
        for b in sites:
            if a.id != b.id:
                out[(a.id, b.id)] = circuity * haversine_km(a.lat, a.lon, b.lat, b.lon)
    return out


class OSRMClient:
    """Minimal client for the OSRM ``table`` service.

    Run your own server (``docker run osrm/osrm-backend``) on a Geofabrik
    extract, or point ``base_url`` at another instance whose usage policy
    allows your volume of requests.
    """

    def __init__(self, base_url: str = "http://localhost:5000", cache_path: str = "osrm_cache.json"):
        self.base_url = base_url.rstrip("/")
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                self.cache = json.load(fh)

    def matrix(self, sites: Sequence) -> dict:
        key = "|".join(f"{s.id}:{s.lat:.5f},{s.lon:.5f}" for s in sites)
        if key not in self.cache:
            import requests  # optional dependency

            coords = ";".join(f"{s.lon},{s.lat}" for s in sites)
            r = requests.get(f"{self.base_url}/table/v1/driving/{coords}",
                             params={"annotations": "distance,duration"}, timeout=60)
            r.raise_for_status()
            self.cache[key] = r.json()["distances"]
            with open(self.cache_path, "w", encoding="utf-8") as fh:
                json.dump(self.cache, fh)
        table = self.cache[key]
        return {(a.id, b.id): table[i][j] / 1000.0
                for i, a in enumerate(sites) for j, b in enumerate(sites) if a.id != b.id}
