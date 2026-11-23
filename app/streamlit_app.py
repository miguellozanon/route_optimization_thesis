"""Interactive dispatch planner.

    pip install -e .[app]
    streamlit run app/streamlit_app.py
"""
import os
import sys
from dataclasses import replace

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brlo.data.instances import iberia_base                      # noqa: E402
from brlo.data.model import Params                                # noqa: E402
from brlo.longhaul.alns import ALNS                               # noqa: E402
from brlo.longhaul.greedy import greedy_plan                      # noqa: E402
from brlo.longhaul.legacy import legacy_plan                      # noqa: E402
from brlo.longhaul.milp import solve_two_phase                    # noqa: E402
from brlo.longhaul.solution import evaluate                       # noqa: E402
from brlo.viz.maps import plan_map_html                           # noqa: E402

st.set_page_config(page_title="BRLO - battery reverse logistics", layout="wide")
st.title("Battery reverse logistics - long-haul dispatch")

with st.sidebar:
    st.header("Scenario")
    scale = st.slider("Monthly volume multiplier", 0.25, 4.0, 1.0, 0.25)
    co2 = st.slider("Carbon price (EUR/t CO2)", 0, 500, 0, 25)
    diesel = st.number_input("Diesel price (EUR/l)", 0.8, 3.0, 1.45, 0.05)
    driver = st.number_input("Driver cost (EUR/h)", 10.0, 60.0, 25.0, 1.0)
    max_stops = st.select_slider("Max stops per trip", [1, 2, 3, 4], 3)
    circuity = st.slider("Road circuity factor", 1.1, 1.5, 1.25, 0.05)
    method = st.radio("Method", ["Two-phase MILP (certified)", "ALNS", "Greedy", "Legacy prototype (2024)"])
    tlim = st.slider("Time limit (s)", 10, 300, 60, 10)
    go = st.button("Optimise", type="primary")

params = Params(diesel_eur_l=diesel, driver_eur_h=driver, co2_price_eur_t=co2, max_stops=max_stops)
inst = iberia_base(stock_scale=scale, circuity=circuity, params=params)

st.subheader("Stock at the depot (kg)")
st.dataframe(pd.DataFrame([inst.depots[0].stock]).round(0), hide_index=True)

if go:
    with st.spinner("Solving..."):
        info = ""
        if method.startswith("Two-phase"):
            r = solve_two_phase(inst, time_limit=tlim)
            plan = r.plan
            info = f"Certified gap: {100 * r.gap:.2f}% (lower bound EUR {r.bound:,.0f}) in {r.seconds:.1f} s"
        elif method == "ALNS":
            a = ALNS(inst, seed=0, eval_time=3).run(greedy_plan(inst), iterations=10_000, time_limit=tlim)
            plan = a.plan
            info = f"{a.evaluations} MILP evaluations in {a.seconds:.1f} s"
        elif method == "Greedy":
            plan = greedy_plan(inst)
        else:
            L = legacy_plan(inst, 1)
            plan = L.plan
            info = f"The prototype would have REPORTED EUR {L.reported_cost:,.0f}"
    ev = evaluate(inst, plan)
    c = st.columns(5)
    c[0].metric("Total cost", f"EUR {ev.cost:,.0f}")
    c[1].metric("Transport", f"EUR {ev.transport:,.0f}")
    c[2].metric("Gate fees", f"EUR {ev.gate_fees:,.0f}")
    c[3].metric("CO2", f"{ev.co2_kg / 1000:,.2f} t")
    c[4].metric("Trips", ev.n_trips)
    if info:
        st.info(info)
    if not ev.feasible:
        st.error("Infeasible plan:\n\n" + "\n".join(f"- {v}" for v in ev.violations[:20]))
    components.html(plan_map_html(inst, plan, title=method), height=620, scrolling=False)
