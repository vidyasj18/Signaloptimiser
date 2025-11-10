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
from datetime import datetime

PCU = {"N": 2542.0, "S": 2760.0, "E": 0.0, "W": 1500.0}
# PCU = {"N": 2880.0, "S": 2760.0, "E": 1560.0, "W": 3480.0}

# Attempt to load totals from outputs/intersection_summary.json
def load_pcu_from_summary(path: str = 'outputs/intersection_summary.json'):
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

# Helper: Webster cycle from total flow and capacity
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

# ============================================================================
# REAL-WORLD PATTERN FUNCTIONS
# ============================================================================

def get_time_of_day_factor(hour: int, rng) -> float:
    """Peak hours (7-9 AM, 5-7 PM) have higher demand and variability"""
    if 7 <= hour < 9:  # Morning peak
        return 1.3 + rng.uniform(-0.2, 0.3)
    elif 17 <= hour < 19:  # Evening peak
        return 1.25 + rng.uniform(-0.15, 0.25)
    elif 22 <= hour or hour < 6:  # Night (low traffic)
        return 0.4 + rng.uniform(-0.1, 0.1)
    else:  # Off-peak
        return 0.9 + rng.uniform(-0.15, 0.15)

def get_weather_factor(weather: str, rng) -> float:
    """Weather affects driver behavior and effective capacity"""
    factors = {
        'clear': 1.0,
        'rain': 0.85 + rng.uniform(-0.05, 0.05),
        'heavy_rain': 0.70 + rng.uniform(-0.1, 0.05),
        'fog': 0.80 + rng.uniform(-0.1, 0.05),
        'snow': 0.60 + rng.uniform(-0.15, 0.1)
    }
    return factors.get(weather, 1.0)

def get_event_factor(has_event: bool, rng) -> float:
    """Special events cause demand spikes"""
    if has_event:
        return 1.4 + rng.uniform(0.0, 0.3)  # 40-70% increase
    return 1.0

def get_day_of_week_factor(day: int, rng) -> float:
    """Weekends have different patterns"""
    if day in [5, 6]:  # Friday, Saturday
        return 1.1 + rng.uniform(-0.1, 0.2)
    elif day == 0:  # Sunday
        return 0.8 + rng.uniform(-0.1, 0.1)
    return 1.0

def apply_real_world_skew(base_pcu: float, hour: int, weather: str, 
                          has_event: bool, day: int, approach: str, rng) -> float:
    """Apply multiple real-world factors to base PCU"""
    # Base variation
    pcu = base_pcu * (1.0 + rng.uniform(-0.1, 0.1))
    
    # Time-of-day effect
    pcu *= get_time_of_day_factor(hour, rng)
    
    # Weather effect (affects all approaches)
    pcu *= get_weather_factor(weather, rng)
    
    # Event effect
    pcu *= get_event_factor(has_event, rng)
    
    # Day-of-week effect
    pcu *= get_day_of_week_factor(day, rng)
    
    # Directional bias (e.g., morning rush: more N->S, evening: more S->N)
    if hour in [7, 8] and approach in ['N', 'E']:  # Morning: inbound to city
        pcu *= (1.1 + rng.uniform(0.0, 0.15))
    elif hour in [17, 18] and approach in ['S', 'W']:  # Evening: outbound
        pcu *= (1.1 + rng.uniform(0.0, 0.15))
    
    return max(100.0, pcu)  # Minimum threshold

def calculate_real_world_delay(c: float, g: float, q: float, s: float, 
                               weather: str) -> float:
    """Calculate delay with real-world adjustments"""
    # Base Webster delay
    if s <= 0 or c <= 0 or g <= 0:
        return float('inf')
    
    X = q / s
    if X >= 1:
        return float('inf')
    
    uniform_delay = (c * (1 - g / c) ** 2) / (2 * (1 - X * (g / c)))
    random_delay = (X ** 2) / (2 * q * (1 - X)) if X > 0 and (1 - X) > 0 else 0.0
    base_delay = uniform_delay + random_delay
    
    # Real-world adjustments: weather increases delay
    weather_multiplier = {
        'clear': 1.0,
        'rain': 1.15,
        'heavy_rain': 1.30,
        'fog': 1.20,
        'snow': 1.50
    }.get(weather, 1.0)
    
    return base_delay * weather_multiplier

# ============================================================================
# DATA LOADING (Flexible: synthetic now, real data later)
# ============================================================================

def generate_synthetic_data(num_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic training data with real-world patterns and skews"""
    rng = np.random.default_rng(seed)
    rows = []
    
    for _ in range(num_samples):
        # Random contextual factors
        hour = rng.integers(0, 24)
        weather = rng.choice(['clear', 'rain', 'heavy_rain', 'fog', 'snow'], 
                            p=[0.6, 0.2, 0.05, 0.1, 0.05])
        has_event = rng.random() < 0.1  # 10% chance of special event
        day = rng.integers(0, 7)  # 0=Sunday, 6=Saturday
        
        # Base PCUs (before real-world skew)
        base_N = float(rng.integers(100, 4000))
        base_S = float(rng.integers(100, 4000))
        base_E = float(rng.integers(100, 4000))
        base_W = float(rng.integers(100, 4000))
        
        # Apply real-world skews
        N = apply_real_world_skew(base_N, hour, weather, has_event, day, 'N', rng)
        S = apply_real_world_skew(base_S, hour, weather, has_event, day, 'S', rng)
        E = apply_real_world_skew(base_E, hour, weather, has_event, day, 'E', rng)
        W = apply_real_world_skew(base_W, hour, weather, has_event, day, 'W', rng)
        
        total = N + S + E + W
        
        # Capacity affected by weather
        weather_capacity_factor = get_weather_factor(weather, rng)
        capacity = 4 * DEFAULT_LANES * SAT_PER_LANE * weather_capacity_factor
        
        # Base Webster cycle
        C_base = webster_cycle_from_capacity(total, capacity)
        
        # Real-world adjustments to cycle (events may need longer cycles)
        if has_event:
            C_base *= (1.0 + rng.uniform(0.05, 0.15))
        
        C = float(np.clip(C_base, 60.0, 180.0))
        
        # Split greens
        greens, eff = split_greens(C, N, S, E, W)
        
        # Calculate actual delay (with real-world factors)
        weather_cap = DEFAULT_LANES * SAT_PER_LANE * weather_capacity_factor
        avg_delay_N = calculate_real_world_delay(C, greens["N"], N, weather_cap, weather) if N > 0 else 0.0
        avg_delay_S = calculate_real_world_delay(C, greens["S"], S, weather_cap, weather) if S > 0 else 0.0
        avg_delay_E = calculate_real_world_delay(C, greens["E"], E, weather_cap, weather) if E > 0 else 0.0
        avg_delay_W = calculate_real_world_delay(C, greens["W"], W, weather_cap, weather) if W > 0 else 0.0
        
        rows.append({
            "N": N, "S": S, "E": E, "W": W,
            "NS": N + S, "EW": E + W,
            "cycle": C,
            "gN": greens["N"], "gS": greens["S"], "gE": greens["E"], "gW": greens["W"],
            # Contextual features (same format as real data would have)
            "hour": hour,
            "weather_clear": 1.0 if weather == 'clear' else 0.0,
            "weather_rain": 1.0 if weather == 'rain' else 0.0,
            "weather_heavy_rain": 1.0 if weather == 'heavy_rain' else 0.0,
            "weather_fog": 1.0 if weather == 'fog' else 0.0,
            "weather_snow": 1.0 if weather == 'snow' else 0.0,
            "has_event": 1.0 if has_event else 0.0,
            "day_of_week": day,
            "is_weekend": 1.0 if day in [0, 5, 6] else 0.0,
            # Target: actual delays (for future optimization)
            "delay_N": avg_delay_N,
            "delay_S": avg_delay_S,
            "delay_E": avg_delay_E,
            "delay_W": avg_delay_W,
            "total_delay": avg_delay_N + avg_delay_S + avg_delay_E + avg_delay_W
        })
    
    return pd.DataFrame(rows)

def load_real_data(filepath: str) -> pd.DataFrame:
    """
    Load real-world traffic data from CSV/JSON.
    Expected columns: N, S, E, W, hour, weather, has_event, day_of_week, cycle, gN, gS, gE, gW
    This function can be implemented when real data is available.
    """
    # TODO: Implement when real data is available
    # Example structure:
    # if filepath.endswith('.csv'):
    #     df = pd.read_csv(filepath)
    # elif filepath.endswith('.json'):
    #     df = pd.read_json(filepath)
    # return df
    raise NotImplementedError("Real data loading not yet implemented. Use synthetic data for now.")

def load_training_data(use_real_data: bool = False, real_data_path: str = None) -> pd.DataFrame:
    """
    Flexible data loading: synthetic or real data.
    Returns DataFrame with consistent structure.
    """
    if use_real_data and real_data_path:
        try:
            return load_real_data(real_data_path)
        except Exception as e:
            print(f"Warning: Could not load real data: {e}. Falling back to synthetic data.")
    
    print("Generating synthetic training data with real-world patterns...")
    return generate_synthetic_data(num_samples=3000, seed=42)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def extract_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract/ensure contextual features are present.
    Handles missing features gracefully (for real data compatibility).
    """
    # Ensure required features exist, fill with defaults if missing
    if 'hour' not in df.columns:
        df['hour'] = 12  # Default: noon
    if 'has_event' not in df.columns:
        df['has_event'] = 0.0
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = 0.0
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = 1  # Default: Monday
    
    # Weather features (one-hot encoded)
    weather_cols = ['weather_clear', 'weather_rain', 'weather_heavy_rain', 'weather_fog', 'weather_snow']
    for col in weather_cols:
        if col not in df.columns:
            df[col] = 1.0 if col == 'weather_clear' else 0.0
    
    return df

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_models(df: pd.DataFrame):
    """Train ML models on training data (synthetic or real)"""
    df = extract_context_features(df)
    
    # 2a) Cycle prediction with context
    cycle_features = ["NS", "EW", "hour", "has_event", "is_weekend", 
                      "weather_rain", "weather_heavy_rain", "weather_fog", "weather_snow"]
    X_cycle = df[cycle_features].values
    y_cycle = df["cycle"].values
    cycle_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15)
    cycle_model.fit(X_cycle, y_cycle)
    
    # 2b) Green time prediction with context
    green_features = ["N", "S", "E", "W", "hour", "has_event", "is_weekend",
                      "weather_rain", "weather_heavy_rain", "weather_fog", "weather_snow",
                      "NS", "EW"]
    X_greens = df[green_features].values
    y_greens = df[["gN", "gS", "gE", "gW"]].values
    green_model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=20)
    green_model.fit(X_greens, y_greens)
    
    return cycle_model, green_model, cycle_features, green_features

# ============================================================================
# PREDICTION WITH CONTEXT
# ============================================================================

def get_current_context():
    """
    Get current contextual information.
    In production, this would fetch real-time data.
    For now, uses current time and defaults.
    """
    now = datetime.now()
    return {
        'hour': now.hour,
        'day_of_week': now.weekday(),  # 0=Monday, 6=Sunday
        'weather': 'clear',  # TODO: Get from weather API
        'has_event': False,  # TODO: Get from event calendar
    }

def prepare_prediction_features(N: float, S: float, E: float, W: float, 
                                context: dict, cycle_features: list, green_features: list):
    """Prepare feature vectors for prediction"""
    NS, EW = N + S, E + W
    hour = context.get('hour', 12)
    has_event = 1.0 if context.get('has_event', False) else 0.0
    day = context.get('day_of_week', 1)
    is_weekend = 1.0 if day in [5, 6] else 0.0
    
    weather = context.get('weather', 'clear')
    weather_features = {
        'clear': [1.0, 0.0, 0.0, 0.0, 0.0],
        'rain': [0.0, 1.0, 0.0, 0.0, 0.0],
        'heavy_rain': [0.0, 0.0, 1.0, 0.0, 0.0],
        'fog': [0.0, 0.0, 0.0, 1.0, 0.0],
        'snow': [0.0, 0.0, 0.0, 0.0, 1.0]
    }
    w_clear, w_rain, w_heavy, w_fog, w_snow = weather_features.get(weather, [1.0, 0.0, 0.0, 0.0, 0.0])
    
    # Cycle features
    X_cycle = np.array([[NS, EW, hour, has_event, is_weekend, w_rain, w_heavy, w_fog, w_snow]])
    
    # Green features
    X_greens = np.array([[N, S, E, W, hour, has_event, is_weekend, w_rain, w_heavy, w_fog, w_snow, NS, EW]])
    
    return X_cycle, X_greens

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Load PCU inputs
loaded = load_pcu_from_summary()
if loaded:
    PCU.update(loaded)
    print(f"Loaded PCU from outputs/intersection_summary.json (cwd={os.getcwd()}): {PCU}")
else:
    print(f"Using fallback PCU (cwd={os.getcwd()}): {PCU}")

# Load training data (synthetic for now, real data later)
df = load_training_data(use_real_data=False)

# Train models
print("Training ML models with contextual features...")
cycle_model, green_model, cycle_features, green_features = train_models(df)
print("Model training complete.")

# Get current context for prediction
context = get_current_context()
print(f"\nPrediction context: hour={context['hour']}, day={context['day_of_week']}, "
      f"weather={context['weather']}, event={context['has_event']}")

# Prepare prediction inputs
N = float(PCU.get("N", 0.0))
S = float(PCU.get("S", 0.0))
E = float(PCU.get("E", 0.0))
W = float(PCU.get("W", 0.0))
present_keys = [k for k in ["N","S","E","W"] if PCU.get(k, 0.0) > 0]
NS, EW = N + S, E + W

# Prepare features
X_cycle_pred, X_greens_pred = prepare_prediction_features(N, S, E, W, context, cycle_features, green_features)

# Predict cycle with context
pred_cycle = float(cycle_model.predict(X_cycle_pred)[0])
pred_cycle = float(np.clip(pred_cycle, 60.0, 180.0))

# Defensive fallback
if pred_cycle <= 0 or not np.isfinite(pred_cycle):
    capacity_present = max(1, len(present_keys)) * DEFAULT_LANES * SAT_PER_LANE
    pred_cycle = webster_cycle_from_capacity(NS + EW, capacity_present)

# Predict greens with context
pred_greens = green_model.predict(X_greens_pred)[0]
gN, gS, gE, gW = [max(0.0, float(x)) for x in pred_greens]

# Normalization so sum greens ≈ effective green
effective = max(0.0, pred_cycle - DEFAULT_LOST_TIME)
sum_g = gN + gS + gE + gW
if sum_g > 0:
    scale = effective / sum_g
    gN, gS, gE, gW = gN * scale, gS * scale, gE * scale, gW * scale

# Build per-approach G/Y/R
def gyr(g, C=pred_cycle, Y=YELLOW):
    y = Y if g > 0 else 0.0
    r = max(0.0, C - (g + y))
    return g, y, r

sched = {"cycle_length": pred_cycle}
if N > 0: sched["NB"] = dict(zip(["green","amber","red"], gyr(gN)))
if S > 0: sched["SB"] = dict(zip(["green","amber","red"], gyr(gS)))
if E > 0: sched["EB"] = dict(zip(["green","amber","red"], gyr(gE)))
if W > 0: sched["WB"] = dict(zip(["green","amber","red"], gyr(gW)))

# Pretty print
print("\n=== Inputs (PCU) ===")
print(PCU)
print("\n=== Predicted Cycle (ML with context) ===")
print(f"{sched['cycle_length']:.2f} s")
print("\n=== Per-approach timings (s) ===")
for k in [x for x in ["NB","SB","EB","WB"] if x in sched]:
    g = sched[k]["green"]; a = sched[k]["amber"]; r = sched[k]["red"]
    print(f"{k}: green={g:.2f}, amber={a:.2f}, red={r:.2f}")

# Save JSON output for comparison script
output_data = {
    "method": "ML",
    "cycle_length": pred_cycle,
    "approaches": {}
}
# Map approach codes
approach_map = {"N": "NB", "S": "SB", "E": "EB", "W": "WB"}
greens_map = {"N": gN, "S": gS, "E": gE, "W": gW}
for appr_code, appr_name in approach_map.items():
    if PCU.get(appr_code, 0.0) > 0:
        output_data["approaches"][appr_name] = {
            "effective_green": greens_map[appr_code],
            "arrival_flow_rate": float(PCU.get(appr_code, 0.0)),  # PCU/hr
            "saturation_flow_rate": DEFAULT_LANES * SAT_PER_LANE,  # PCU/hr
            "green": sched[appr_name]["green"] if appr_name in sched else 0.0,
            "amber": sched[appr_name]["amber"] if appr_name in sched else 0.0,
            "red": sched[appr_name]["red"] if appr_name in sched else 0.0
        }

os.makedirs('outputs', exist_ok=True)
output_path = 'outputs/ml_signal_plan.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
print(f"\n=== Saved ML signal plan to {output_path} ===")

# Phase-based diagram with exclusivity and all-red
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
