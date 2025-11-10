import json
import os
import math
from typing import Any, Dict, Optional


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


# Internal helper to normalise PCU maps
def _normalise_pcu_dict(pcu: Dict[str, float]) -> Dict[str, float]:
    """Return dict with N/S/E/W keys, accepting NB/SB/etc forms."""
    result = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
    mapping = {
        "N": "N", "NB": "N", "NORTH": "N",
        "S": "S", "SB": "S", "SOUTH": "S",
        "E": "E", "EB": "E", "EAST": "E",
        "W": "W", "WB": "W", "WEST": "W",
    }
    for key, value in (pcu or {}).items():
        norm_key = mapping.get(key.upper()) if isinstance(key, str) else None
        if norm_key is not None:
            try:
                result[norm_key] = float(value)
            except Exception:
                result[norm_key] = 0.0
    return result


def compute_webster_plan(
    pcu_override: Optional[Dict[str, float]] = None,
    save_json: bool = True,
    output_path: str = 'outputs/webster_signal_plan.json',
    ensure_outputs_dir: bool = True,
) -> Dict:
    """
    Compute Webster plan using optional PCU override.

    Returns the plan dictionary (same structure as saved JSON) and writes to file if save_json is True.
    """
    # Fallback PCU if JSON missing/empty (edit as needed)
    fallback_pcu = {"N": 2880.0, "S": 2760.0, "E": 1560.0, "W": 3480.0}

    if pcu_override is not None:
        pcu = _normalise_pcu_dict(pcu_override)
    else:
        pcu = load_pcu_from_summary()

    if not any(pcu.values()):
        pcu = fallback_pcu
        print(f"Using fallback PCU (cwd={os.getcwd()}): {pcu}")
    else:
        print(f"Loaded PCU (cwd={os.getcwd()}): {pcu}")

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
    if N > 0:
        sched["NB"] = gyr(gN)
    if S > 0:
        sched["SB"] = gyr(gS)
    if E > 0:
        sched["EB"] = gyr(gE)
    if W > 0:
        sched["WB"] = gyr(gW)

    output_data = {
        "method": "Webster",
        "cycle_length": cycle,
        "approaches": {}
    }

    approach_map = {"N": "NB", "S": "SB", "E": "EB", "W": "WB"}
    greens_map = {"N": gN, "S": gS, "E": gE, "W": gW}
    for appr_code, appr_name in approach_map.items():
        if pcu.get(appr_code, 0.0) > 0 and appr_name in sched:
            output_data["approaches"][appr_name] = {
                "effective_green": greens_map[appr_code],
                "arrival_flow_rate": float(pcu.get(appr_code, 0.0)),  # PCU/hr
                "saturation_flow_rate": DEFAULT_LANES * SAT_PER_LANE,  # PCU/hr
                "green": sched[appr_name]["green"],
                "amber": sched[appr_name]["amber"],
                "red": sched[appr_name]["red"]
            }

    if ensure_outputs_dir:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    if save_json:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Saved Webster signal plan to {output_path}")

    output_data["_diagnostics"] = {
        "pcu": pcu,
        "effective_green_total": effective_green,
        "g_NS": g_NS,
        "g_EW": g_EW,
        "present_approaches": [k for k, v in pcu.items() if v > 0]
    }
    return output_data


def create_phase_diagram(plan: Dict) -> Any:
    """Return a Plotly figure representing the Webster phase diagram."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("plotly is required for creating the phase diagram") from exc

    cycle = plan.get("cycle_length", 0.0)
    approaches = plan.get("approaches", {})

    def get_group_green(group_keys):
        """Get the maximum green time within a phase group (NS or EW)"""
        greens = [approaches.get(k, {}).get("green", 0.0) for k in group_keys if approaches.get(k)]
        return max(greens) if greens else 0.0

    # Use max green time within each group (not sum, not average)
    # NS green = max(NB green, SB green)
    # EW green = max(EB green, WB green)
    g_ns = get_group_green(["NB", "SB"])
    g_ew = get_group_green(["EB", "WB"])
    
    # Ensure we're using exact values (round to 2 decimals to match table display)
    g_ns = round(g_ns, 2)
    g_ew = round(g_ew, 2)
    
    # Validate: these should match the maximum green time shown in the per-approach table
    # For example: if NB=20.77 and SB=19.91, then g_ns should be 20.77
    # If EB=11.25 and WB=25.10, then g_ew should be 25.10

    fig = go.Figure()
    t = 0.0  # Global time tracker
    colors = {"green": "#2ecc71", "amber": "#f1c40f", "red": "#e74c3c"}
    seen_legend = set()

    # Phase 1: NS Green (EW is Red)
    if g_ns > 0:
        # NS Green
        fig.add_trace(go.Bar(x=[g_ns], y=["NS"], orientation="h",
                             marker_color=colors["green"], name="green", base=t,
                             showlegend="green" not in seen_legend))
        if "green" not in seen_legend:
            seen_legend.add("green")
        # EW Red (while NS is green)
        fig.add_trace(go.Bar(x=[g_ns], y=["EW"], orientation="h",
                             marker_color=colors["red"], name="red", base=t,
                             showlegend="red" not in seen_legend))
        if "red" not in seen_legend:
            seen_legend.add("red")
        t += g_ns
        
        # NS Amber (EW still Red)
        fig.add_trace(go.Bar(x=[YELLOW], y=["NS"], orientation="h",
                             marker_color=colors["amber"], name="amber", base=t,
                             showlegend="amber" not in seen_legend))
        if "amber" not in seen_legend:
            seen_legend.add("amber")
        fig.add_trace(go.Bar(x=[YELLOW], y=["EW"], orientation="h",
                             marker_color=colors["red"], showlegend=False, base=t))
        t += YELLOW
    else:
        # If no NS green, both are red for the cycle
        fig.add_trace(go.Bar(x=[cycle], y=["NS"], orientation="h",
                             marker_color=colors["red"], name="red", base=0.0,
                             showlegend="red" not in seen_legend))
        if "red" not in seen_legend:
            seen_legend.add("red")
        fig.add_trace(go.Bar(x=[cycle], y=["EW"], orientation="h",
                             marker_color=colors["red"], showlegend=False, base=0.0))
        t = cycle
    
    # Phase 2: All-Red (both NS and EW are red)
    if g_ns > 0 and g_ew > 0 and ALL_RED > 0:
        fig.add_trace(go.Bar(x=[ALL_RED], y=["NS"], orientation="h",
                             marker_color=colors["red"], showlegend=False, base=t))
        fig.add_trace(go.Bar(x=[ALL_RED], y=["EW"], orientation="h",
                             marker_color=colors["red"], showlegend=False, base=t))
        t += ALL_RED
    
    # Phase 3: EW Green (NS is Red)
    if g_ew > 0:
        # EW Green
        fig.add_trace(go.Bar(x=[g_ew], y=["EW"], orientation="h",
                             marker_color=colors["green"], showlegend=False, base=t))
        # NS Red (while EW is green)
        fig.add_trace(go.Bar(x=[g_ew], y=["NS"], orientation="h",
                             marker_color=colors["red"], showlegend=False, base=t))
        t += g_ew
        
        # EW Amber (NS still Red)
        fig.add_trace(go.Bar(x=[YELLOW], y=["EW"], orientation="h",
                             marker_color=colors["amber"], showlegend=False, base=t))
        fig.add_trace(go.Bar(x=[YELLOW], y=["NS"], orientation="h",
                             marker_color=colors["red"], showlegend=False, base=t))
        t += YELLOW
    else:
        # If no EW green but NS exists, NS stays red
        if g_ns > 0:
            remaining = cycle - t
            if remaining > 0:
                fig.add_trace(go.Bar(x=[remaining], y=["NS"], orientation="h",
                                     marker_color=colors["red"], showlegend=False, base=t))
                fig.add_trace(go.Bar(x=[remaining], y=["EW"], orientation="h",
                                     marker_color=colors["red"], showlegend=False, base=t))
        t = cycle
    
    # Phase 4: Final All-Red (both NS and EW are red until cycle ends)
    if t < cycle:
        final_all_red = cycle - t
        if final_all_red > 0:
            fig.add_trace(go.Bar(x=[final_all_red], y=["NS"], orientation="h",
                                 marker_color=colors["red"], showlegend=False, base=t))
            fig.add_trace(go.Bar(x=[final_all_red], y=["EW"], orientation="h",
                                 marker_color=colors["red"], showlegend=False, base=t))

    fig.update_layout(
        barmode="stack",
        title=f"Phase Timeline (NS vs EW) — Cycle={cycle:.1f}s",
        xaxis_title="Time (s)",
        yaxis_title="Phase Group",
        height=320,
    )
    return fig


def main() -> None:
    plan = compute_webster_plan()

    print("=== Inputs (PCU) ===")
    print(plan["_diagnostics"]["pcu"])
    print("\n=== Computed (Webster) ===")
    cycle = plan["cycle_length"]
    effective_green = plan["_diagnostics"]["effective_green_total"]
    g_NS = plan["_diagnostics"]["g_NS"]
    g_EW = plan["_diagnostics"]["g_EW"]
    print(f"Cycle: {cycle:.2f} s  |  Lost time: {DEFAULT_LOST_TIME:.2f} s  |  Eff. green: {effective_green:.2f} s")
    print(f"NS share: {g_NS:.2f} s,  EW share: {g_EW:.2f} s")
    print("\n=== Per-approach timings (s) ===")
    for appr, data in plan["approaches"].items():
        g = data.get("green", 0.0)
        a = data.get("amber", 0.0)
        r = data.get("red", 0.0)
        print(f"{appr}: green={g:.2f}, amber={a:.2f}, red={r:.2f}")

    print("\n=== Phase sequence (exclusive, with all-red) ===")
    if g_NS > 0:
        print(f"NS: green {g_NS:.2f}s -> amber {YELLOW:.2f}s")
    if g_NS > 0 and g_EW > 0 and ALL_RED > 0:
        print(f"All-red: {ALL_RED:.2f}s")
    if g_EW > 0:
        print(f"EW: green {g_EW:.2f}s -> amber {YELLOW:.2f}s")
    if g_NS > 0 and g_EW > 0 and ALL_RED > 0:
        print(f"All-red: {ALL_RED:.2f}s")

    try:
        fig = create_phase_diagram(plan)
        fig.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()


