# Inputs source: try to read outputs/intersection_summary.json; fall back to hardcoded PCUs
import json
import os


# Constants
DEFAULT_LANES = 2           # lanes per approach (used for cycle load)
SAT_PER_LANE = 1800         # PCU/hr/lane
DEFAULT_LOST_TIME = 12.0    # s
YELLOW = 3.0                # s

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# PCU = {"N": 2542.0, "S": 2760.0, "E": 0.0, "W": 1500.0}
PCU = {"N": 2880.0, "S": 2760.0, "E": 1560.0, "W": 3480.0}

# Attempt to load totals from outputs/intersection_summary.json
def load_pcu_from_summary(path: str = 'intersection_summary.json'):
    try:
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        # Map NB/SB/EB/WB -> N/S/E/W using 'total_pcu'
        mapping = {"NB": "N", "SB": "S", "EB": "E", "WB": "W"}
        pcu = {m: 0.0 for m in mapping.values()}
        for k_raw, v in summary.items():
            key = mapping.get(k_raw)
            if key is None:
                continue
            try:
                total_pcu = float(v.get('total_pcu', 0.0))
            except Exception:
                total_pcu = 0.0
            pcu[key] = total_pcu
        return pcu
    except Exception:
        return None

loaded = load_pcu_from_summary()
if loaded:
    PCU.update(loaded)
    print(f"Loaded PCU from outputs/intersection_summary.json (cwd={os.getcwd()}): {PCU}")
else:
    print(f"Using fallback PCU (cwd={os.getcwd()}): {PCU}")

# Helper: Webster cycle from total flow and capacity (synthetic ground truth)
def webster_cycle_from_capacity(total_flow, capacity, L=DEFAULT_LOST_TIME):
    if capacity <= 0:
        return 0.0
    Y = total_flow / capacity
    Y = min(Y, 0.95)                     # avoid infinite cycles
    C = (1.5 * L + 5.0) / (1.0 - Y)      # classic Webster
    return float(np.clip(C, 60.0, 180.0))

# Helper: split effective green between NS and EW, then to approaches
def split_greens(C, N, S, E, W, L=DEFAULT_LOST_TIME):
    NS, EW = N + S, E + W
    eff = max(0.0, C - L)
    if (NS + EW) <= 0:
        return dict(N=0, S=0, E=0, W=0), eff
    g_NS = eff * (NS / (NS + EW))
    g_EW = eff - g_NS
    gN = g_NS * (N / NS) if NS > 0 else 0.0
    gS = g_NS * (S / NS) if NS > 0 else 0.0
    gE = g_EW * (E / EW) if EW > 0 else 0.0
    gW = g_EW * (W / EW) if EW > 0 else 0.0
    return dict(N=gN, S=gS, E=gE, W=gW), eff

# 1) Create synthetic training data using Webster rules
rng = np.random.default_rng(42)
rows = []
for _ in range(2000):
    # random PCUs per approach (reasonably wide range)
    N = float(rng.integers(100, 4000))
    S = float(rng.integers(100, 4000))
    E = float(rng.integers(100, 4000))
    W = float(rng.integers(100, 4000))
    total = N + S + E + W
    # Assume all 4 approaches present in synthetic generation
    capacity = 4 * DEFAULT_LANES * SAT_PER_LANE
    C = webster_cycle_from_capacity(total, capacity)
    greens, eff = split_greens(C, N, S, E, W)
    rows.append({
        "N": N, "S": S, "E": E, "W": W,
        "NS": N + S, "EW": E + W,
        "cycle": C,
        "gN": greens["N"], "gS": greens["S"], "gE": greens["E"], "gW": greens["W"],
    })

df = pd.DataFrame(rows)

# 2) Train ML models
# 2a) Cycle from NS/EW (linear)
X_cycle = df[["NS", "EW"]].values
y_cycle = df["cycle"].values
cycle_model = LinearRegression().fit(X_cycle, y_cycle)

# 2b) Greens per approach from approach PCUs (nonlinear)
X_greens = df[["N", "S", "E", "W"]].values
y_greens = df[["gN", "gS", "gE", "gW"]].values
green_model = RandomForestRegressor(n_estimators=300, random_state=42)
green_model.fit(X_greens, y_greens)

# 3) Predict for your hardcoded PCUs (auto-handle missing approaches e.g., T-junction)
N = float(PCU.get("N", 0.0))
S = float(PCU.get("S", 0.0))
E = float(PCU.get("E", 0.0))
W = float(PCU.get("W", 0.0))
present_keys = [k for k in ["N","S","E","W"] if PCU.get(k, 0.0) > 0]
NS, EW = N + S, E + W

# Cycle: prefer ML estimate on NS/EW, then clamp
pred_cycle = float(cycle_model.predict(np.array([[NS, EW]]))[0])
pred_cycle = float(np.clip(pred_cycle, 60.0, 180.0))

# Defensive: recompute via Webster from capacity of present approaches if ML yields edge case
if pred_cycle <= 0 or not np.isfinite(pred_cycle):
    capacity_present = max(1, len(present_keys)) * DEFAULT_LANES * SAT_PER_LANE
    pred_cycle = webster_cycle_from_capacity(NS + EW, capacity_present)

# Greens per approach via ML
pred_greens = green_model.predict(np.array([[N, S, E, W]]))[0]
gN, gS, gE, gW = [max(0.0, float(x)) for x in pred_greens]

# Optional normalization so sum greens ≈ effective green
effective = max(0.0, pred_cycle - DEFAULT_LOST_TIME)
sum_g = gN + gS + gE + gW
if sum_g > 0:
    scale = effective / sum_g
    gN, gS, gE, gW = gN * scale, gS * scale, gE * scale, gW * scale

# 4) Build per-approach G/Y/R (only include present approaches)
def gyr(g, C=pred_cycle, Y=YELLOW):
    y = Y if g > 0 else 0.0
    r = max(0.0, C - (g + y))
    return g, y, r

sched = {"cycle_length": pred_cycle}
if N > 0: sched["NB"] = dict(zip(["green","amber","red"], gyr(gN)))
if S > 0: sched["SB"] = dict(zip(["green","amber","red"], gyr(gS)))
if E > 0: sched["EB"] = dict(zip(["green","amber","red"], gyr(gE)))
if W > 0: sched["WB"] = dict(zip(["green","amber","red"], gyr(gW)))

# 5) Pretty print
print("=== Inputs (PCU) ===")
print(PCU)
print("\n=== Predicted Cycle ===")
print(f"{sched['cycle_length']:.2f} s")
print("\n=== Per-approach timings (s) ===")
for k in [x for x in ["NB","SB","EB","WB"] if x in sched]:
    g = sched[k]["green"]; a = sched[k]["amber"]; r = sched[k]["red"]
    print(f"{k}: green={g:.2f}, amber={a:.2f}, red={r:.2f}")

# 6) Phase-based diagram with exclusivity and all-red
try:
    import plotly.graph_objects as go

    C = sched["cycle_length"]
    colors = {"green":"#2ecc71", "amber":"#f1c40f", "red":"#e74c3c"}

    # Aggregate predicted greens into phase greens (only present approaches)
    g_NS = (gN if N > 0 else 0.0) + (gS if S > 0 else 0.0)
    g_EW = (gE if E > 0 else 0.0) + (gW if W > 0 else 0.0)
    ALL_RED = 2.0  # seconds of all-red between phases

    # Build phases in time order: NS green->amber, all-red, EW green->amber, all-red
    phases = [
        ("NS", "green", g_NS if g_NS > 0 else 0.0),
        ("NS", "amber", YELLOW if g_NS > 0 else 0.0),
        ("BOTH", "red", ALL_RED if (g_NS > 0 and g_EW > 0) else 0.0),
        ("EW", "green", g_EW if g_EW > 0 else 0.0),
        ("EW", "amber", YELLOW if g_EW > 0 else 0.0),
        ("BOTH", "red", ALL_RED if (g_NS > 0 and g_EW > 0) else 0.0),
    ]

    # Render timelines: only show rows for non-empty groups
    fig = go.Figure()
    t = 0.0
    seen_legend = set()
    for actor, kind, dur in phases:
        if dur <= 0:
            continue
        # Determine legend visibility once per color/kind
        show = kind not in seen_legend
        seen_legend.add(kind)

        if actor == "NS" and g_NS > 0:
            # NS active (green/amber), EW must be red for same duration
            fig.add_trace(go.Bar(x=[dur], y=["NS"], orientation="h",
                                 marker_color=colors[kind], name=kind,
                                 showlegend=show, base=t))
            if g_EW > 0:
                fig.add_trace(go.Bar(x=[dur], y=["EW"], orientation="h",
                                     marker_color=colors["red"], name="red",
                                     showlegend=("red" not in seen_legend), base=t))
                seen_legend.add("red")
        elif actor == "EW" and g_EW > 0:
            # EW active (green/amber), NS must be red for same duration
            fig.add_trace(go.Bar(x=[dur], y=["EW"], orientation="h",
                                 marker_color=colors[kind], name=kind,
                                 showlegend=show, base=t))
            if g_NS > 0:
                fig.add_trace(go.Bar(x=[dur], y=["NS"], orientation="h",
                                     marker_color=colors["red"], name="red",
                                     showlegend=("red" not in seen_legend), base=t))
                seen_legend.add("red")
        else:  # BOTH (all-red)
            if g_NS > 0:
                fig.add_trace(go.Bar(x=[dur], y=["NS"], orientation="h",
                                     marker_color=colors["red"], name="red",
                                     showlegend=("red" not in seen_legend), base=t))
                seen_legend.add("red")
            if g_EW > 0:
                fig.add_trace(go.Bar(x=[dur], y=["EW"], orientation="h",
                                     marker_color=colors["red"], name="red",
                                     showlegend=False, base=t))
        t += dur

    # If the built phases don't fill whole cycle (due to normalization), append trailing red on present groups
    if t < C:
        tail = C - t
        if g_NS > 0:
            fig.add_trace(go.Bar(x=[tail], y=["NS"], orientation="h", marker_color=colors["red"], name="red", showlegend=False, base=t))
        if g_EW > 0:
            fig.add_trace(go.Bar(x=[tail], y=["EW"], orientation="h", marker_color=colors["red"], name="red", showlegend=False, base=t))

    fig.update_layout(
        barmode="stack",
        title=f"Phase Timeline (NS vs EW) — Cycle={C:.1f}s",
        xaxis_title="Time (s)",
        yaxis_title="Phase Group",
        height=320,
    )
    fig.show()
except Exception:
    pass