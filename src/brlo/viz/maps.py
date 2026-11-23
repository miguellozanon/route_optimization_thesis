"""Interactive Leaflet map of a long-haul plan (self-contained HTML, OpenStreetMap tiles).

Trips are drawn as straight segments between sites unless an OSRM geometry
provider is plugged in; the popup of each trip lists vehicle, stops and loads.
"""

from __future__ import annotations

import html
import json

from ..data.model import LongHaulInstance
from ..longhaul.solution import Plan, evaluate

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html,body{{margin:0;height:100%;font:14px/1.4 system-ui,sans-serif;color:#0b0b0b;background:#fcfcfb}}
 #map{{position:absolute;top:0;bottom:0;left:340px;right:0}}
 #side{{position:absolute;top:0;bottom:0;left:0;width:340px;overflow:auto;padding:16px;box-sizing:border-box;border-right:1px solid #e5e4e0}}
 h1{{font-size:17px;margin:0 0 4px}} .muted{{color:#52514e;font-size:12px}}
 table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:10px}}
 td,th{{padding:3px 4px;border-bottom:1px solid #eee;text-align:left}} td.n{{text-align:right;font-variant-numeric:tabular-nums}}
 .sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}}
 .trip{{cursor:pointer}} .trip:hover{{background:#f1f0ec}}
</style></head><body>
<div id="side"><h1>{title}</h1><div class="muted">{subtitle}</div>
<table><tr><th>KPI</th><th class="n">value</th></tr>{kpis}</table>
<table><tr><th>#</th><th>vehicle</th><th>stops</th><th class="n">kg</th></tr>{trips}</table>
<p class="muted">Straight segments between sites; costs use road distance = great-circle &times; circuity.
Map data &copy; OpenStreetMap contributors.</p></div>
<div id="map"></div>
<script>
const data = {data};
const map = L.map('map');
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{maxZoom: 18, attribution: '&copy; OpenStreetMap contributors'}}).addTo(map);
const bounds = [];
data.sites.forEach(s => {{
  const m = L.circleMarker([s.lat, s.lon], {{radius: s.kind==='depot'?9:7, color:'#fcfcfb', weight:2,
     fillColor: s.kind==='depot' ? '#0b0b0b' : (s.used ? '#2a78d6' : '#a8a7a2'), fillOpacity: 1}}).addTo(map);
  m.bindPopup('<b>'+s.name+'</b><br>'+s.info);
  bounds.push([s.lat, s.lon]);
}});
const layers = [];
data.trips.forEach((t, i) => {{
  const pl = L.polyline(t.path, {{color: t.color, weight: 3, opacity: 0.85, dashArray: t.dash}}).addTo(map);
  pl.bindPopup(t.popup); layers.push(pl);
}});
document.querySelectorAll('.trip').forEach(row => row.addEventListener('click', () => {{
  const pl = layers[+row.dataset.i]; map.fitBounds(pl.getBounds(), {{padding:[40,40]}}); pl.openPopup();
}}));
map.fitBounds(bounds, {{padding: [30, 30]}});
</script></body></html>"""


def plan_map_html(inst: LongHaulInstance, plan: Plan, title: str = "Battery long-haul dispatch",
                  subtitle: str = "") -> str:
    ev = evaluate(inst, plan)
    used = {j for t in plan.trips for j in t.stops}
    sites = []
    for d in inst.depots:
        stock = ", ".join(f"{k}: {v / 1000:.1f} t" for k, v in d.stock.items() if v > 0)
        sites.append({"name": d.name, "lat": d.lat, "lon": d.lon, "kind": "depot", "used": True,
                      "info": "Storage depot<br>" + html.escape(stock)})
    for f in inst.facilities:
        fees = ", ".join(f"{k}: {v:+.2f} &euro;/kg" for k, v in f.fees.items())
        sites.append({"name": f.name, "lat": f.lat, "lon": f.lon, "kind": f.kind, "used": f.id in used,
                      "info": f"{f.kind} &middot; {f.country}<br>{fees}"})
    pos = {s.id: (s.lat, s.lon) for s in [*inst.depots, *inst.facilities]}
    vehicles = sorted({t.vehicle for t in plan.trips})
    vcol = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(vehicles)}
    trips, rows = [], []
    for i, t in enumerate(plan.trips):
        path = [pos[t.depot], *[pos[j] for j in t.stops], pos[t.depot]]
        loads = "<br>".join(f"{t.stops[s]}: {k} {q / 1000:.2f} t ({t.containers.get((s, k), 0)} cont.)"
                            for (s, k), q in sorted(t.deliveries.items()))
        v = inst.V[t.vehicle]
        trips.append({"path": path, "color": vcol[t.vehicle], "dash": "6 6" if v.energy == "electric" else None,
                      "popup": f"<b>Trip {i + 1}</b> &middot; {html.escape(v.name)}<br>{loads}"})
        rows.append(f'<tr class="trip" data-i="{i}"><td>{i + 1}</td><td><span class="sw" style="background:{vcol[t.vehicle]}"></span>'
                    f'{t.vehicle}</td><td>{" &rarr; ".join(t.stops)}</td><td class="n">{t.cargo_kg():,.0f}</td></tr>')
    kpi = [("Total cost", f"&euro; {ev.cost:,.0f}"), ("Transport", f"&euro; {ev.transport:,.0f}"),
           ("Gate fees", f"&euro; {ev.gate_fees:,.0f}"), ("CO2", f"{ev.co2_kg / 1000:,.2f} t"),
           ("Distance", f"{ev.distance_km:,.0f} km"), ("Trips", str(ev.n_trips)),
           ("Unshipped", f"{ev.unshipped_kg:,.0f} kg"), ("Feasible", "yes" if ev.feasible else "NO")]
    kpis = "".join(f'<tr><td>{k}</td><td class="n">{v}</td></tr>' for k, v in kpi)
    return TEMPLATE.format(title=html.escape(title), subtitle=html.escape(subtitle), kpis=kpis,
                           trips="".join(rows), data=json.dumps({"sites": sites, "trips": trips}))
