import json
import os
import math
from typing import Dict


# Constants (tune per site/IRC tables)
DEFAULT_LANES = 2          # lanes per approach used for capacity
SAT_PER_LANE = 1800        # PCU/hr/lane (typical design value)
DEFAULT_LOST_TIME = 12.0   # s per cycle (startup + change intervals)
YELLOW = 3.0               # s amber per phase
ALL_RED = 2.0              # s all-red between phases


def load_pcu_from_summary(path: str = 'outputs/intersection_summary.json') -> Dict[str, float]:
    """Load approach PCU totals from app JSON; map NB/SB/EB/WB -> N/S/E/W.
    Missing approaches are returned as 0.0.
    """
    mapping = {"NB": "N", "SB": "S", "EB": "E", "WB": "W"}
    pcu = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
    if not os.path.exists(path):
        return pcu
    try:
        with open(path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        for k_raw, v in summary.items():
            key = mapping.get(k_raw)
            if key is None:
                continue
            try:
                pcu[key] = float(v.get('total_pcu', 0.0))
            except Exception:
                pcu[key] = 0.0
    except Exception:
        pass
    return pcu


def webster_cycle_from_capacity(total_flow: float, capacity: float, lost_time: float) -> float:
    if capacity <= 0:
        return 0.0
    Y = total_flow / capacity
    Y = min(Y, 0.95)  # avoid infinite cycles
    C = (1.5 * lost_time + 5.0) / (1.0 - Y)
    return float(max(60.0, min(C, 180.0)))


def main() -> None:
    # Fallback PCU if JSON missing/empty (edit as needed)
    fallback_pcu = {"N": 500.0, "S": 450.0, "E": 600.0, "W": 550.0}

    pcu = load_pcu_from_summary()
    if not any(pcu.values()):
        pcu = fallback_pcu
        print(f"Using fallback PCU (cwd={os.getcwd()}): {pcu}")
    else:
        print(f"Loaded PCU from outputs/intersection_summary.json (cwd={os.getcwd()}): {pcu}")
    N, S, E, W = pcu['N'], pcu['S'], pcu['E'], pcu['W']
    present = {k: v for k, v in pcu.items() if v > 0}
    num_present = len(present)

    total_flow = N + S + E + W
    capacity = num_present * DEFAULT_LANES * SAT_PER_LANE
    cycle = webster_cycle_from_capacity(total_flow, capacity, DEFAULT_LOST_TIME)
    effective_green = max(0.0, cycle - DEFAULT_LOST_TIME)

    NS = N + S
    EW = E + W

    if NS + EW > 0:
        g_NS = effective_green * (NS / (NS + EW))
    else:
        g_NS = 0.0
    g_EW = effective_green - g_NS

    # Split within phase groups proportionally to approach shares
    gN = g_NS * (N / NS) if NS > 0 and N > 0 else 0.0
    gS = g_NS * (S / NS) if NS > 0 and S > 0 else 0.0
    gE = g_EW * (E / EW) if EW > 0 and E > 0 else 0.0
    gW = g_EW * (W / EW) if EW > 0 and W > 0 else 0.0

    def gyr(g: float) -> Dict[str, float]:
        a = YELLOW if g > 0 else 0.0
        r = max(0.0, cycle - (g + a))
        return {"green": g, "amber": a, "red": r}

    sched = {"cycle_length": cycle}
    if N > 0: sched["NB"] = gyr(gN)
    if S > 0: sched["SB"] = gyr(gS)
    if E > 0: sched["EB"] = gyr(gE)
    if W > 0: sched["WB"] = gyr(gW)

    # Output
    print("=== Inputs (PCU) ===")
    print(pcu)
    print("\n=== Computed (Webster) ===")
    print(f"Cycle: {cycle:.2f} s  |  Lost time: {DEFAULT_LOST_TIME:.2f} s  |  Eff. green: {effective_green:.2f} s")
    print(f"NS share: {g_NS:.2f} s,  EW share: {g_EW:.2f} s")
    print("\n=== Per-approach timings (s) ===")
    for k in [x for x in ["NB","SB","EB","WB"] if x in sched]:
        g = sched[k]["green"]; a = sched[k]["amber"]; r = sched[k]["red"]
        print(f"{k}: green={g:.2f}, amber={a:.2f}, red={r:.2f}")

    # Phase sequence (textual):
    print("\n=== Phase sequence (exclusive, with all-red) ===")
    if g_NS > 0:
        print(f"NS: green {g_NS:.2f}s → amber {YELLOW if g_NS>0 else 0:.2f}s")
    if g_NS > 0 and g_EW > 0 and ALL_RED > 0:
        print(f"All-red: {ALL_RED:.2f}s")
    if g_EW > 0:
        print(f"EW: green {g_EW:.2f}s → amber {YELLOW if g_EW>0 else 0:.2f}s")
    if g_NS > 0 and g_EW > 0 and ALL_RED > 0:
        print(f"All-red: {ALL_RED:.2f}s")

    # Phase diagram (like final.py) if plotly is available
    try:
        import plotly.graph_objects as go

        colors = {"green": "#2ecc71", "amber": "#f1c40f", "red": "#e74c3c"}

        # Build phases in time order: NS green->amber, all-red, EW green->amber, all-red
        phases = [
            ("NS", "green", g_NS if g_NS > 0 else 0.0),
            ("NS", "amber", YELLOW if g_NS > 0 else 0.0),
            ("BOTH", "red", ALL_RED if (g_NS > 0 and g_EW > 0) else 0.0),
            ("EW", "green", g_EW if g_EW > 0 else 0.0),
            ("EW", "amber", YELLOW if g_EW > 0 else 0.0),
            ("BOTH", "red", ALL_RED if (g_NS > 0 and g_EW > 0) else 0.0),
        ]

        fig = go.Figure()
        t = 0.0
        seen_legend = set()
        for actor, kind, dur in phases:
            if dur <= 0:
                continue
            show = kind not in seen_legend
            seen_legend.add(kind)

            if actor == "NS" and g_NS > 0:
                fig.add_trace(go.Bar(x=[dur], y=["NS"], orientation="h",
                                     marker_color=colors[kind], name=kind,
                                     showlegend=show, base=t))
                if g_EW > 0:
                    fig.add_trace(go.Bar(x=[dur], y=["EW"], orientation="h",
                                         marker_color=colors["red"], name="red",
                                         showlegend=("red" not in seen_legend), base=t))
                    seen_legend.add("red")
            elif actor == "EW" and g_EW > 0:
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

        # Fill any trailing time in cycle as red on present groups
        if t < cycle:
            tail = cycle - t
            if g_NS > 0:
                fig.add_trace(go.Bar(x=[tail], y=["NS"], orientation="h",
                                     marker_color=colors["red"], name="red",
                                     showlegend=False, base=t))
            if g_EW > 0:
                fig.add_trace(go.Bar(x=[tail], y=["EW"], orientation="h",
                                     marker_color=colors["red"], name="red",
                                     showlegend=False, base=t))

        fig.update_layout(
            barmode="stack",
            title=f"Phase Timeline (NS vs EW) — Cycle={cycle:.1f}s",
            xaxis_title="Time (s)",
            yaxis_title="Phase Group",
            height=320,
        )
        fig.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()


