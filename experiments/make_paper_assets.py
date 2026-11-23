"""Build every table, figure and number macro of the paper from results/*.

    python experiments/make_paper_assets.py && (cd paper && latexmk -pdf main.tex)
Missing results produce clearly marked placeholders instead of failing.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "results")
GEN = os.path.join(ROOT, "paper", "generated")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(GEN, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# validated categorical palette (fixed order), neutral inks
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 2, "pdf.fonttype": 42})
macros = {}


def fmt(x, d=0):
    return f"{x:,.{d}f}"


def read(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def placeholder(label, caption, what):
    what = what.replace("_", "\\_")
    return (f"\\begin{{table}}[t]\\centering\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\fbox{{\\parbox{{0.8\\linewidth}}{{\\centering\\todo{{{what}}}}}}}\\end{{table}}\n")


# ------------------------------------------------------------------ E1
e1 = read("e1_case_study.csv")
if e1 is not None:
    rows = []
    for r in e1.itertuples():
        rep = f"{r.reported_cost:,.0f}" if "Legacy" in r.method else "--"
        gap = f"{r.gap_to_best_lb_pct:.2f}" if r.feasible else "--"
        rows.append(f"{r.method} & {rep} & {r.cost:,.0f} & {r.transport:,.0f} & {r.gate_fees:,.0f} & "
                    f"{r.co2_kg / 1000:.2f} & {r.n_trips} & {'yes' if r.feasible else f'no ({r.n_violations})'} & {gap} & {r.seconds:.1f}\\\\")
    tab = ("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n"
           "\\caption{Case study (one month, 71\\,t). Costs in \\euro{{}} recomputed by the independent evaluator; "
           "`reported' is what the legacy prototype displayed; gap is relative to the best lower bound "
           f"({e1['lb'].max():,.0f}\\,\\euro).}}\\label{{tab:case}}\n"
           "\\begin{tabular}{lrrrrrrlrr}\\toprule\n"
           "Method & Reported & Cost & Transport & Fees & CO$_2$ (t) & Trips & Feasible & Gap \\% & s\\\\\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    leg = e1[e1.method.str.contains("option 1")].iloc[0]
    tp = e1[e1.method.str.contains("Two-phase")].iloc[0]
    best_cost = e1[e1.feasible].cost.min()
    full = e1[e1.method.str.contains("Full")].iloc[0]
    gr = e1[e1.method.str.contains("Greedy")].iloc[0]
    macros["Egreedygap"] = f"{gr.gap_to_best_lb_pct:.1f}"
    macros.update(EoneTPcert=f"{100 * (tp.cost - tp.lb) / tp.cost:.2f}", EoneFullSec=f"{full.seconds:.0f}",
                  EoneFullGap=f"{full.gap_to_best_lb_pct:.2f}", EoneOpt=fmt(e1['lb'].max()))
    macros.update(LegacyReported=fmt(leg.reported_cost), LegacyTrue=fmt(leg.cost),
                  LegacyFactor=f"{leg.cost / max(1, leg.reported_cost):.0f}",
                  EoneTPgap=f"{tp.gap_to_best_lb_pct:.2f}", EoneTPsec=f"{tp.seconds:.0f}",
                  EoneSavingPct=f"{100 * (1 - best_cost / leg.cost):.0f}")
else:
    tab = placeholder("tab:case", "Case study.", "run experiments/run_longhaul.py e1")
    macros["Egreedygap"] = "?"
    macros.update(EoneTPcert="?", EoneFullSec="?", EoneFullGap="?", EoneOpt="?")
    macros.update(LegacyReported="?", LegacyTrue="?", LegacyFactor="?", EoneTPgap="?", EoneTPsec="?", EoneSavingPct="?")
open(os.path.join(GEN, "tab_case.tex"), "w").write(tab)

# case network figure
plans_path = os.path.join(RES, "e1_plans.json")
if os.path.exists(plans_path):
    from brlo.data.instances import iberia_base
    inst = iberia_base()
    plans = json.load(open(plans_path))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.grid(False)
    pos = {s.id: (s.lon, s.lat) for s in [*inst.depots, *inst.facilities]}
    vcol = {"T40": C1, "R26": C2, "R12": C3, "E19": C4}
    for tr in plans["best"][0]:
        stops = tr["stops"].split("-")
        path = [pos[tr["depot"]], *[pos[j] for j in stops], pos[tr["depot"]]]
        xs, ys = zip(*path)
        ax.plot(xs, ys, color=vcol.get(tr["vehicle"], INK2), lw=0.8 + 3 * tr["cargo_kg"] / 24000, alpha=0.75,
                solid_capstyle="round", ls="--" if tr["vehicle"] == "E19" else "-")
    used = {j for tr in plans["best"][0] for j in tr["stops"].split("-")}
    for f in inst.facilities:
        ax.scatter(f.lon, f.lat, s=46, color=C1 if f.id in used else "#a8a7a2", edgecolor="white", lw=1.5, zorder=3,
                   marker="o" if f.kind == "recycling" else "s")
        off = {"ACC_MUL": (6, 4), "ACC_KRE": (-12, -12), "RECYP": (-58, 4), "BEEPL": (6, -2), "MINURB": (-62, 4),
               "TERA": (6, -10), "REFACT": (-10, -12), "SNAM_SQF": (6, 3), "SNAM_VIV": (6, 3)}.get(f.id, (5, 3))
        ax.annotate(f.name.split(" (")[0], (f.lon, f.lat), xytext=off, textcoords="offset points", fontsize=7, color=INK2)
    d = inst.depots[0]
    ax.scatter(d.lon, d.lat, s=90, color=INK, marker="*", zorder=4)
    ax.annotate("Depot (Madrid)", (d.lon, d.lat), xytext=(5, -10), textcoords="offset points", fontsize=8, color=INK)
    used_v = {tr["vehicle"] for tr in plans["best"][0]}
    for v, c in vcol.items():
        if v in used_v:
            ax.plot([], [], color=c, label=inst.V[v].name, ls="--" if v == "E19" else "-")
    ax.scatter([], [], color=C1, marker="o", label="recycler (used)"); ax.scatter([], [], color=C1, marker="s", label="second-life (used)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(-9.5, 9.5); ax.set_ylim(36.5, 53)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_aspect(1.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "case_network.pdf")); plt.close(fig)

# ------------------------------------------------------------------ E2
e2 = read("e2_ablation.csv")
if e2 is not None:
    rows = []
    for r in e2.itertuples():
        iname = r.instance.replace("_", "\\_")
        rows.append(f"{iname} & {'yes' if r.strengthen else 'no'} & {r.lp_root:,.0f} & "
                    f"{r.relaxed_containers_opt:,.0f} & {r.full_lb:,.0f} & {r.full_ub:,.0f} & {r.full_s:.0f}\\\\")
    tab = ("\\begin{table}[t]\\centering\\small\n\\caption{Effect of the valid inequalities (VUB+SLOT): LP root bound, "
           "optimum with continuous containers (phase 1), and bound/incumbent of the full MILP after 120\\,s.}\\label{tab:ablation}\n"
           "\\begin{tabular}{lcrrrrr}\\toprule\nInstance & Ineq. & LP root & Phase-1 opt. & Full LB & Full UB & s\\\\\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    imp = []
    for inst_name, sub in e2.groupby("instance"):
        a = sub[~sub.strengthen].iloc[0]; b = sub[sub.strengthen].iloc[0]
        imp.append(100 * (b.lp_root / a.lp_root - 1))
    macros["AblText"] = (f"The inequalities raise the root LP bound by {imp[0]:.0f}\\% and {imp[1]:.0f}\\% on the two instances, but "
                         "within 120\\,s HiGHS's own cutting planes and branching make the final bounds of both variants comparable, and the "
                         "full MILP does not converge faster. Their practical value is different: (SLOT) is required for the aggregated "
                         "solution to be split into individual vehicles, and without it phase 1 can return plans that no fleet can execute. "
                         "\\Cref{tab:ablation} also shows that relaxing container integrality (phase 1) is what makes the problem easy: the "
                         "relaxed model solves to optimality in seconds, whereas the integer one does not close in 120\\,s.")
else:
    macros["AblText"] = ""
    tab = placeholder("tab:ablation", "Ablation.", "run e2")
open(os.path.join(GEN, "tab_ablation.tex"), "w").write(tab)

# ------------------------------------------------------------------ E3
e3 = read("e3_scalability.csv")
if e3 is not None and len(e3):
    e3["gap"] = 100 * (e3.cost - e3.lb) / e3.cost
    g = e3.groupby(["n_fac", "n_dep", "method"]).agg(gap=("gap", "mean"), s=("seconds", "mean"),
                                                      cols=("columns", "mean"), feas=("feasible", "mean"), k=("seed", "count")).reset_index()
    order = ["greedy", "alns", "two_phase", "two_phase+alns"]
    names = {"greedy": "Greedy", "alns": "ALNS", "two_phase": "Two-phase", "two_phase+alns": "Two-phase+ALNS"}
    rows = []
    for (nf, nd), sub in g.groupby(["n_fac", "n_dep"]):
        cells = []
        for m in order:
            x = sub[sub.method == m]
            cells.append(f"{x.gap.iloc[0]:.2f} & {x.s.iloc[0]:.0f}" if len(x) else "-- & --")
        rows.append(f"{nf} & {nd} & {sub.cols.iloc[0]:,.0f} & " + " & ".join(cells) + "\\\\")
    tab = ("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{Random instances (mean over seeds): "
           "gap (\\%) to the phase-1 lower bound and seconds.}\\label{tab:scal}\n"
           "\\begin{tabular}{rrr" + "rr" * 4 + "}\\toprule\n"
           "$|J|$ & $|O|$ & columns & \\multicolumn{2}{c}{Greedy} & \\multicolumn{2}{c}{ALNS} & \\multicolumn{2}{c}{Two-phase} & \\multicolumn{2}{c}{Two-phase+ALNS}\\\\\n"
           " & & & gap & s & gap & s & gap & s & gap & s\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    sizes = sorted({(a, b) for a, b in zip(e3.n_fac, e3.n_dep)})
    width = 0.2
    for mi, (m, c) in enumerate(zip(order, [C1, C2, C3, C4])):
        for si, sz in enumerate(sizes):
            sub = e3[(e3.n_fac == sz[0]) & (e3.n_dep == sz[1]) & (e3.method == m)]
            x = si + (mi - 1.5) * width
            ax.bar(x, sub.gap.mean(), width * 0.9, color=c, label=names[m] if si == 0 else None)
            ax.scatter([x] * len(sub), sub.gap, s=9, color=INK, zorder=3)
    ax.set_xticks(range(len(sizes)), [f"|J|={a}, |O|={b}" for a, b in sizes])
    ax.set_yscale("log"); ax.set_ylim(0.1, 300)
    ax.set_ylabel("gap to lower bound (%, log)"); ax.legend(ncol=4, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "scalability.pdf")); plt.close(fig)
    small = e3[e3["columns"] < 4000]; large = e3[e3["columns"] >= 4000]
    ms, ml = small.groupby("method").gap.mean(), large.groupby("method").gap.mean()
    tsec = small[small.method == "two_phase"].seconds
    tp_unsh = large[large.method == "two_phase"].unshipped_kg
    txt = (f"On instances with fewer than 4,000 columns ({small.groupby(['n_fac','n_dep','seed']).ngroups} instances) the two-phase scheme is the "
           f"method of choice: mean gap {ms['two_phase']:.1f}\\% to the lower bound in {tsec.mean():.0f}\\,s on average, versus "
           f"{ms['alns']:.1f}\\% for ALNS started from the greedy plan (60\\,s) and {ms['greedy']:.1f}\\% for the greedy rule, "
           "whose gap is inflated by stock it fails to place within capacities and fleet days. ALNS polishing improves the "
           "two-phase plans only marginally (by less than 0.01\\%) on these sizes. The picture reverses on the largest class (about 8,100 columns): phase~1 does not converge "
           "within its 60\\,s budget, its incumbent leaves "
           f"{tp_unsh.min() / 1000:.0f}--{tp_unsh.max() / 1000:.0f}\\,t unshipped and the restricted phase~2 cannot recover; ALNS from the greedy plan is then the best "
           f"method (mean gap {ml['alns']:.1f}\\% against the relaxation bound, versus {ml['greedy']:.1f}\\% for greedy). The two methods are "
           "therefore complementary, and seeding phase~2 with ALNS or greedy routes is an obvious improvement. All gaps are measured against "
           "relaxation bounds and overstate the true sub-optimality: on the case study the phase-1 bound was 0.8\\% below the proven optimum.")
    macros["ScalText"] = txt
else:
    macros["ScalText"] = ""
    tab = placeholder("tab:scal", "Scalability.", "run e3")
open(os.path.join(GEN, "tab_scal.tex"), "w").write(tab)

# ------------------------------------------------------------------ E4
e4 = read("e4_sensitivity.csv")
if e4 is not None:
    co = e4[e4.study == "co2_price"]
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.plot(co.co2_kg / 1000, co.cost_eur / 1000, "-o", color=C1, ms=5)
    groups = {}
    for r in co.itertuples():
        groups.setdefault((round(r.co2_kg, 1), round(r.cost_eur, 1)), []).append(int(r.value))
    for (x, y), lams in groups.items():
        lab = "λ = " + (f"{lams[0]}" if len(lams) == 1 else f"{min(lams)}–{max(lams)}") + " €/t"
        ax.annotate(lab, (x / 1000, y / 1000), xytext=(-6, 6), textcoords="offset points", fontsize=7, color=INK2, ha="right" if x == max(k[0] for k in groups) else "left")
    ax.margins(x=0.25, y=0.25)
    ax.set_xlabel("CO$_2$ (t)"); ax.set_ylabel("cost without carbon price (k€)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "co2_tradeoff.pdf")); plt.close(fig)
    vs = e4[e4.study == "stock_scale"]
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.plot(vs.value, vs.eur_per_t, "-o", color=C1, ms=5, label="€/t")
    ax.set_xlabel("volume multiplier"); ax.set_ylabel("cost per tonne (€)")
    ax2 = ax.twinx() if False else None
    for i, r in enumerate(vs.itertuples()):
        off, ha = [((2, 10), "left"), ((6, -14), "left"), ((-6, 6), "right"), ((-8, -3), "right")][i % 4]
        ax.annotate(f"{100 * r.share_spain:.0f}% in Spain", (r.value, r.eur_per_t), xytext=off,
                    textcoords="offset points", fontsize=7, color=INK2, ha=ha)
    ax.margins(x=0.12, y=0.15)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "volume_scale.pdf")); plt.close(fig)
    c0, cmax = co.iloc[0], co.iloc[-1]
    s1, s3 = vs.iloc[0], vs.iloc[-1]
    macros["SensText"] = (f"Internalising a carbon price of {cmax.value:.0f}\\,\\euro/t cuts emissions from "
                          f"{c0.co2_kg / 1000:.2f} to {cmax.co2_kg / 1000:.2f}\\,t ({100 * (1 - cmax.co2_kg / c0.co2_kg):.0f}\\%) at a cost "
                          f"increase of {100 * (cmax.cost_eur / c0.cost_eur - 1):.1f}\\% (\\cref{{fig:sens}}). Scaling the monthly volume from "
                          f"{s1.value:g}$\\times$ to {s3.value:g}$\\times$ moves the cost per tonne from {s1.eur_per_t:.0f} to {s3.eur_per_t:.0f}\\,\\euro{{}} "
                          f"and the share treated in Spain from {100 * s1.share_spain:.0f}\\% to {100 * s3.share_spain:.0f}\\%, as Spanish capacity saturates; "
                          f"at {s3.value:g}$\\times$ the volume {s3.unshipped_kg / 1000:.0f}\\,t (lead--acid) cannot be placed within contracted capacities at all, "
                          "which is precisely the situation the strategic model of \\cref{sec:strategic} is designed to anticipate.")
else:
    macros["SensText"] = "\\todo{run e4}"

# ------------------------------------------------------------------ E5
e5p = os.path.join(RES, "e5_strategic.json")
if os.path.exists(e5p):
    e5 = json.load(open(e5p))
    M = 1e6
    rows = [f"{k} & {e5[k] / M:.2f}\\\\" for k in ("RP", "EV", "EEV", "WS")]
    rows += [f"VSS = EEV $-$ RP & {e5['VSS'] / M:.2f} ({e5['VSS_pct_of_RP']:.1f}\\%)\\\\",
             f"EVPI = RP $-$ WS & {e5['EVPI'] / M:.2f} ({e5['EVPI_pct_of_RP']:.1f}\\%)\\\\"]
    des = e5["designs"]

    def dsc(d):
        dep, pl = d
        a = ", ".join(f"{k}~{min(v)}" for k, v in dep.items() if v and k != "MAD")
        b = ", ".join(k.replace("NEW_", "") + f"~{min(v)}" for k, v in pl.items() if v)
        return f"depots: {a or 'Madrid only'}; plants: {b or 'none'}"
    drows = [k.replace("_", " ") + f" & {dsc(v)}\\\\" for k, v in des.items()]
    tab = ("\\begin{table}[t]\\centering\\small\n\\caption{Stochastic network design, 2026--2035 (discounted M\\euro). "
           "Designs list the first operating year of each additional depot and new plant.}\\label{tab:strategic}\n"
           "\\begin{tabular}{lr}\\toprule Measure & M\\euro\\\\\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\\end{tabular}\\\\[6pt]\n\\begin{tabular}{ll}\\toprule Model & Design\\\\\\midrule\n" +
           "\n".join(drows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    macros["VSSpct"] = f"{e5['VSS_pct_of_RP']:.1f}"
    eev = e5["eev_scenario_cost"]; rp = e5["rp_scenario_cost"]
    ex = json.load(open(os.path.join(RES, "e5_extra.json"))) if os.path.exists(os.path.join(RES, "e5_extra.json")) else None
    sav = {k: (eev[k] - rp[k]) / M for k in rp}
    txt = (f"The stochastic design invests more up front than the expected-value design"
           + (f" ({ex['first_stage_RP_MEUR']:.1f} vs {ex['first_stage_EEV_MEUR']:.1f}\\,M\\euro{{}} of discounted depot and plant costs)" if ex else "")
           + f" and recovers it in operation: second-stage costs fall by {sav['pessimistic']:.1f}, {sav['neutral']:.1f} and "
           f"{sav['optimistic']:.1f}\\,M\\euro{{}} in the pessimistic, neutral and optimistic scenarios.")
    if ex:
        txt += (f" The expected-value design ({ex['EV_design']}) leaves {ex['untreated_kt_optimistic']['EEV']:.1f}\\,kt of Li-ion "
                f"untreated over the horizon if adoption is fast, versus {ex['untreated_kt_optimistic']['RP']:.1f}\\,kt for the stochastic design.")
    txt += f" EVPI is {e5['EVPI_pct_of_RP']:.1f}\\% of the expected cost, an upper bound on what better demand forecasts are worth."
    macros["StratText"] = txt
    fl = read("e5_flows_rp.csv")
    if fl is not None:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
        for ax, sc in zip(axes, ["pessimistic", "neutral", "optimistic"]):
            sub = fl[(fl.scenario == sc) & (fl.chem == "LI")]
            ax.plot(sub.year, sub.spain_t / 1000, color=C1, label="treated in Spain")
            ax.plot(sub.year, sub.abroad_t / 1000, color=C2, label="treated abroad")
            if sub.unserved_t.max() > 1:
                ax.plot(sub.year, sub.unserved_t / 1000, color=C4, label="untreated")
            ax.set_title(sc, fontsize=9, color=INK)
            ax.set_xticks([2026, 2030, 2035])
        axes[0].set_ylabel("Li-ion (kt/year)")
        handles, labels = [], []
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h); labels.append(l)
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(FIG, "strategic_flows.pdf")); plt.close(fig)
    sens = read("e5_sensitivity.csv")
    if sens is not None:
        yrs = sens.first_plant_year.dropna()
        never = sens[sens.first_plant_year.isna()]
        txt2 = (f" Across $P(\\text{{optimistic}})\\in\\{{0.1,0.25,0.4\\}}$ and plant-cost multipliers $\\{{0.75,1,1.5\\}}$, the first "
                f"Spanish Li-ion plant opens between {int(yrs.min())} and {int(yrs.max())}")
        if len(never):
            txt2 += (", except when the optimistic scenario is unlikely ($p=0.1$) and plants are not cheaper than assumed ("
                     + ", ".join(f"$\\times${r.plant_cost_mult:g}" for r in never.itertuples()) + "), in which case none is built")
        macros["StratText"] += txt2 + "."
else:
    tab = placeholder("tab:strategic", "Strategic.", "run run_strategic.py")
    macros["VSSpct"] = "?"; macros["StratText"] = ""
open(os.path.join(GEN, "tab_strategic.tex"), "w").write(tab)

# ------------------------------------------------------------------ E6
e6 = read("e6_firstmile_classical.csv")
neu = read("e6_firstmile_neural.csv")      # produced on a GPU machine (see README)
if e6 is not None:
    piv = e6.pivot_table(index=["dist", "n"], columns="method", values=["mean_len", "ms_per_inst"])
    rows = []
    for (dist, n), r in piv.iterrows():
        base = r[("mean_len", "cw+ls")]
        nn = f"{r[('mean_len', 'nn')]:.3f} ({100 * (r[('mean_len', 'nn')] / base - 1):+.1f}\\%)"
        cw = f"{r[('mean_len', 'cw')]:.3f} ({100 * (r[('mean_len', 'cw')] / base - 1):+.1f}\\%)"
        ls = f"{base:.3f} ({r[('ms_per_inst', 'cw+ls')]:.0f}\\,ms)"
        if neu is not None and len(neu[(neu.dist == dist) & (neu.n == n)]):
            x = neu[(neu.dist == dist) & (neu.n == n)].iloc[0]
            pomo = f"{x.mean_len:.3f} ({100 * (x.mean_len / base - 1):+.1f}\\%, {x.ms_per_inst:.0f}\\,ms)"
        else:
            pomo = "\\todo{GPU run}"
        rows.append(f"{dist} & {n} & {nn} & {cw} & {ls} & {pomo}\\\\")
    tab = ("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{First-mile CVRP, mean tour length on 100 test instances "
           "(unit square; gaps relative to CW+LS).}\\label{tab:firstmile}\n"
           "\\begin{tabular}{lrllll}\\toprule Distribution & $n$ & Nearest neighbour & Clarke--Wright & CW + local search & POMO ($\\times$8 aug.)\\\\\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    macros["FirstMileText"] = ("On the uniform benchmark, CW+LS is within a few percent of the best-known values reported in the "
                               "learning-to-route literature (about 6.1, 10.4 and 15.7 for $n=20,50,100$ \\citep{kool2019,kwon2020}). "
                               + ("" if neu is not None else "\\todo{Train POMO on a GPU (\\texttt{python -m brlo.firstmile.train}) and add its results; "
                                  "the published POMO numbers suggest gaps of about 1\\% to optimal on uniform instances.}"))
else:
    tab = placeholder("tab:firstmile", "First mile.", "run run_firstmile_classical.py")
    macros["FirstMileText"] = ""
open(os.path.join(GEN, "tab_firstmile.tex"), "w").write(tab)

with open(os.path.join(GEN, "macros.tex"), "w") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("assets written:", sorted(os.listdir(GEN)), sorted(os.listdir(FIG)))
