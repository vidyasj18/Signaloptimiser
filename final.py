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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

PCU = {"N": 2542.0, "S": 2760.0, "E": 0.0, "W": 1500.0}
PCU2 = {"N": 2880.0, "S": 2760.0, "E": 1560.0, "W": 3480.0}

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

def generate_and_save_training_csv(csv_path: str = 'outputs/synthetic_training_data.csv', 
                                    num_samples: int = 3000, seed: int = 42, 
                                    force_regenerate: bool = False) -> pd.DataFrame:
    """
    Generate synthetic training CSV once and save it.
    If CSV already exists and force_regenerate=False, load it instead.
    """
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    
    if os.path.exists(csv_path) and not force_regenerate:
        print(f"Loading existing training CSV from {csv_path}")
        return pd.read_csv(csv_path)
    
    print(f"Generating synthetic training data ({num_samples} samples)...")
    df = generate_synthetic_data(num_samples=num_samples, seed=seed)
    df.to_csv(csv_path, index=False)
    print(f"Saved training CSV to {csv_path}")
    return df

def train_models_from_csv(csv_path: str = 'outputs/synthetic_training_data.csv', calculate_metrics=True):
    """
    Load CSV and train ML models.
    Returns (cycle_model, green_model, cycle_features, green_features, metrics)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV not found: {csv_path}. Run generate_and_save_training_csv() first.")
    
    print(f"Loading training data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Training ML models on {len(df)} samples...")
    return train_models(df, calculate_metrics=calculate_metrics)

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

def train_models(df: pd.DataFrame, calculate_metrics=True):
    """
    Train ML models on training data (synthetic or real)
    Returns models, features, and optionally metrics
    """
    df = extract_context_features(df)
    
    # 2a) Cycle prediction with context
    cycle_features = ["NS", "EW", "hour", "has_event", "is_weekend", 
                      "weather_rain", "weather_heavy_rain", "weather_fog", "weather_snow"]
    X_cycle = df[cycle_features].values
    y_cycle = df["cycle"].values
    
    # Split for metrics calculation
    if calculate_metrics and len(df) > 100:
        X_cycle_train, X_cycle_test, y_cycle_train, y_cycle_test = train_test_split(
            X_cycle, y_cycle, test_size=0.2, random_state=42
        )
    else:
        X_cycle_train, X_cycle_test = X_cycle, X_cycle
        y_cycle_train, y_cycle_test = y_cycle, y_cycle
        calculate_metrics = False
    
    cycle_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15)
    cycle_model.fit(X_cycle_train, y_cycle_train)
    
    # Calculate cycle model metrics
    cycle_metrics = None
    if calculate_metrics:
        y_cycle_pred = cycle_model.predict(X_cycle_test)
        cycle_metrics = {
            "r2_score": float(r2_score(y_cycle_test, y_cycle_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_cycle_test, y_cycle_pred))),
            "mae": float(mean_absolute_error(y_cycle_test, y_cycle_pred)),
            "test_size": len(y_cycle_test),
            "train_size": len(y_cycle_train)
        }
    
    # 2b) Green time prediction with context
    green_features = ["N", "S", "E", "W", "hour", "has_event", "is_weekend",
                      "weather_rain", "weather_heavy_rain", "weather_fog", "weather_snow",
                      "NS", "EW"]
    X_greens = df[green_features].values
    y_greens = df[["gN", "gS", "gE", "gW"]].values
    
    # Split for metrics calculation
    if calculate_metrics and len(df) > 100:
        X_greens_train, X_greens_test, y_greens_train, y_greens_test = train_test_split(
            X_greens, y_greens, test_size=0.2, random_state=42
        )
    else:
        X_greens_train, X_greens_test = X_greens, X_greens
        y_greens_train, y_greens_test = y_greens, y_greens
    
    green_model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=20)
    green_model.fit(X_greens_train, y_greens_train)
    
    # Calculate green model metrics
    green_metrics = None
    if calculate_metrics:
        y_greens_pred = green_model.predict(X_greens_test)
        green_metrics = {
            "gN": {
                "r2_score": float(r2_score(y_greens_test[:, 0], y_greens_pred[:, 0])),
                "rmse": float(np.sqrt(mean_squared_error(y_greens_test[:, 0], y_greens_pred[:, 0]))),
                "mae": float(mean_absolute_error(y_greens_test[:, 0], y_greens_pred[:, 0]))
            },
            "gS": {
                "r2_score": float(r2_score(y_greens_test[:, 1], y_greens_pred[:, 1])),
                "rmse": float(np.sqrt(mean_squared_error(y_greens_test[:, 1], y_greens_pred[:, 1]))),
                "mae": float(mean_absolute_error(y_greens_test[:, 1], y_greens_pred[:, 1]))
            },
            "gE": {
                "r2_score": float(r2_score(y_greens_test[:, 2], y_greens_pred[:, 2])),
                "rmse": float(np.sqrt(mean_squared_error(y_greens_test[:, 2], y_greens_pred[:, 2]))),
                "mae": float(mean_absolute_error(y_greens_test[:, 2], y_greens_pred[:, 2]))
            },
            "gW": {
                "r2_score": float(r2_score(y_greens_test[:, 3], y_greens_pred[:, 3])),
                "rmse": float(np.sqrt(mean_squared_error(y_greens_test[:, 3], y_greens_pred[:, 3]))),
                "mae": float(mean_absolute_error(y_greens_test[:, 3], y_greens_pred[:, 3]))
            },
            "test_size": len(y_greens_test),
            "train_size": len(y_greens_train)
        }
    
    # Feature importance
    cycle_feature_importance = dict(zip(cycle_features, cycle_model.feature_importances_.tolist()))
    green_feature_importance = dict(zip(green_features, green_model.feature_importances_.tolist()))
    
    metrics = {
        "cycle_model": cycle_metrics,
        "green_model": green_metrics,
        "cycle_feature_importance": cycle_feature_importance,
        "green_feature_importance": green_feature_importance,
        "cycle_features": cycle_features,
        "green_features": green_features
    } if calculate_metrics else None
    
    return cycle_model, green_model, cycle_features, green_features, metrics

# ============================================================================
# SAVE METRICS AND GENERATE VISUALIZATIONS
# ============================================================================

def save_model_metrics(metrics: dict, output_dir: str = 'outputs'):
    """Save model performance metrics to JSON file"""
    if not metrics:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, 'ml_model_performance.json')
    
    # Prepare metrics for JSON serialization
    metrics_to_save = {
        "cycle_model": metrics.get("cycle_model"),
        "green_model": metrics.get("green_model"),
        "cycle_feature_importance": metrics.get("cycle_feature_importance"),
        "green_feature_importance": metrics.get("green_feature_importance"),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"Saved model metrics to {metrics_file}")
    return metrics_file

def generate_model_visualizations(metrics: dict, models: tuple, df: pd.DataFrame, 
                                 output_dir: str = 'report/images/ml_model_performance'):
    """
    Generate visualizations for ML model performance
    models: (cycle_model, green_model, cycle_features, green_features)
    """
    if not metrics:
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    cycle_model, green_model, cycle_features, green_features = models
    
    # Prepare data for visualization
    df = extract_context_features(df)
    
    # Split data for predictions
    X_cycle = df[cycle_features].values
    y_cycle = df["cycle"].values
    X_cycle_train, X_cycle_test, y_cycle_train, y_cycle_test = train_test_split(
        X_cycle, y_cycle, test_size=0.2, random_state=42
    )
    
    X_greens = df[green_features].values
    y_greens = df[["gN", "gS", "gE", "gW"]].values
    X_greens_train, X_greens_test, y_greens_train, y_greens_test = train_test_split(
        X_greens, y_greens, test_size=0.2, random_state=42
    )
    
    generated_files = []
    
    # 1. Cycle Model: Prediction vs Actual Scatter Plot
    y_cycle_pred = cycle_model.predict(X_cycle_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_cycle_test, y_cycle_pred, alpha=0.5)
    plt.plot([y_cycle_test.min(), y_cycle_test.max()], 
             [y_cycle_test.min(), y_cycle_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Cycle Length (s)')
    plt.ylabel('Predicted Cycle Length (s)')
    if metrics.get("cycle_model"):
        r2 = metrics["cycle_model"]["r2_score"]
        rmse = metrics["cycle_model"]["rmse"]
        plt.title(f'Cycle Model: Prediction vs Actual\nR² = {r2:.4f}, RMSE = {rmse:.2f}s')
    else:
        plt.title('Cycle Model: Prediction vs Actual')
    plt.grid(True, alpha=0.3)
    cycle_scatter_file = os.path.join(output_dir, 'cycle_prediction_scatter.png')
    plt.savefig(cycle_scatter_file, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(cycle_scatter_file)
    
    # 2. Cycle Model: Feature Importance
    if metrics.get("cycle_feature_importance"):
        importance_dict = metrics["cycle_feature_importance"]
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]
        features_sorted = [features[i] for i in sorted_idx[:10]]  # Top 10
        importances_sorted = [importances[i] for i in sorted_idx[:10]]
        
        plt.figure(figsize=(10, 6))
        plt.barh(features_sorted, importances_sorted)
        plt.xlabel('Feature Importance')
        plt.title('Cycle Model: Top 10 Feature Importance')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        cycle_importance_file = os.path.join(output_dir, 'cycle_feature_importance.png')
        plt.savefig(cycle_importance_file, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(cycle_importance_file)
    
    # 3. Green Split Model: Prediction vs Actual (4 subplots)
    y_greens_pred = green_model.predict(X_greens_test)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    approaches = ['gN', 'gS', 'gE', 'gW']
    
    for idx, approach in enumerate(approaches):
        ax = axes[idx]
        y_actual = y_greens_test[:, idx]
        y_pred = y_greens_pred[:, idx]
        ax.scatter(y_actual, y_pred, alpha=0.5)
        ax.plot([y_actual.min(), y_actual.max()], 
                [y_actual.min(), y_actual.max()], 'r--', lw=2)
        ax.set_xlabel(f'Actual {approach} (s)')
        ax.set_ylabel(f'Predicted {approach} (s)')
        if metrics.get("green_model") and approach in metrics["green_model"]:
            gm = metrics["green_model"][approach]
            ax.set_title(f'{approach}: R² = {gm["r2_score"]:.4f}, RMSE = {gm["rmse"]:.2f}s')
        else:
            ax.set_title(f'{approach}: Prediction vs Actual')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    green_scatter_file = os.path.join(output_dir, 'green_split_prediction_scatter.png')
    plt.savefig(green_scatter_file, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(green_scatter_file)
    
    # 4. Green Split Model: Feature Importance
    if metrics.get("green_feature_importance"):
        importance_dict = metrics["green_feature_importance"]
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        sorted_idx = np.argsort(importances)[::-1]
        features_sorted = [features[i] for i in sorted_idx[:10]]
        importances_sorted = [importances[i] for i in sorted_idx[:10]]
        
        plt.figure(figsize=(10, 6))
        plt.barh(features_sorted, importances_sorted)
        plt.xlabel('Feature Importance')
        plt.title('Green Split Model: Top 10 Feature Importance')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        green_importance_file = os.path.join(output_dir, 'green_split_feature_importance.png')
        plt.savefig(green_importance_file, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(green_importance_file)
    
    print(f"Generated {len(generated_files)} visualization files in {output_dir}")
    return generated_files

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
# PREDICTION FUNCTION
# ============================================================================

def predict_ml_signal_plan(pcu_dict: dict, cycle_model, green_model, 
                           cycle_features: list, green_features: list,
                           output_path: str = None) -> dict:
    """
    Predict ML signal plan for given PCU dictionary.
    
    Args:
        pcu_dict: Dictionary with keys "N", "S", "E", "W" and PCU values
        cycle_model: Trained cycle prediction model
        green_model: Trained green time prediction model
        cycle_features: List of feature names for cycle model
        green_features: List of feature names for green model
        output_path: Optional path to save JSON output
    
    Returns:
        Dictionary with signal plan (same format as webster.py output)
    """
    # Normalize PCU dict
    pcu = {"N": float(pcu_dict.get("N", 0.0)), 
           "S": float(pcu_dict.get("S", 0.0)),
           "E": float(pcu_dict.get("E", 0.0)),
           "W": float(pcu_dict.get("W", 0.0))}
    
    # Get current context for prediction
    context = get_current_context()
    
    # Prepare prediction inputs
    N, S, E, W = pcu["N"], pcu["S"], pcu["E"], pcu["W"]
    present_keys = [k for k in ["N","S","E","W"] if pcu.get(k, 0.0) > 0]
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
    
    # Build output data structure
    output_data = {
        "method": "ML",
        "cycle_length": pred_cycle,
        "approaches": {}
    }
    
    # Map approach codes
    approach_map = {"N": "NB", "S": "SB", "E": "EB", "W": "WB"}
    greens_map = {"N": gN, "S": gS, "E": gE, "W": gW}
    for appr_code, appr_name in approach_map.items():
        if pcu.get(appr_code, 0.0) > 0:
            output_data["approaches"][appr_name] = {
                "effective_green": greens_map[appr_code],
                "arrival_flow_rate": float(pcu.get(appr_code, 0.0)),  # PCU/hr
                "saturation_flow_rate": DEFAULT_LANES * SAT_PER_LANE,  # PCU/hr
                "green": sched[appr_name]["green"] if appr_name in sched else 0.0,
                "amber": sched[appr_name]["amber"] if appr_name in sched else 0.0,
                "red": sched[appr_name]["red"] if appr_name in sched else 0.0
            }
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Saved ML signal plan to {output_path}")
    
    return output_data

# ============================================================================
# MAIN EXECUTION (for backward compatibility)
# ============================================================================

if __name__ == "__main__":
    # Load PCU inputs
    loaded = load_pcu_from_summary()
    if loaded:
        PCU.update(loaded)
        print(f"Loaded PCU from outputs/intersection_summary.json (cwd={os.getcwd()}): {PCU}")
    else:
        print(f"Using fallback PCU (cwd={os.getcwd()}): {PCU}")
    
    # Generate and save training CSV
    csv_path = 'outputs/synthetic_training_data.csv'
    df = generate_and_save_training_csv(csv_path)
    
    # Train models
    print("Training ML models with contextual features...")
    cycle_model, green_model, cycle_features, green_features, metrics = train_models_from_csv(csv_path)
    print("Model training complete.")
    
    if metrics:
        print("\n=== Model Performance Metrics ===")
        if metrics["cycle_model"]:
            print(f"Cycle Model - R²: {metrics['cycle_model']['r2_score']:.4f}, RMSE: {metrics['cycle_model']['rmse']:.2f}s, MAE: {metrics['cycle_model']['mae']:.2f}s")
        if metrics["green_model"]:
            print("Green Split Model:")
            for approach in ["gN", "gS", "gE", "gW"]:
                if approach in metrics["green_model"]:
                    gm = metrics["green_model"][approach]
                    print(f"  {approach} - R²: {gm['r2_score']:.4f}, RMSE: {gm['rmse']:.2f}s, MAE: {gm['mae']:.2f}s")
    
    # Predict signal plan
    output_path = 'outputs/ml_signal_plan.json'
    plan = predict_ml_signal_plan(PCU, cycle_model, green_model, cycle_features, green_features, output_path)
    
    # Pretty print
    print("\n=== Inputs (PCU) ===")
    print(PCU)
    print("\n=== Predicted Cycle (ML with context) ===")
    print(f"{plan['cycle_length']:.2f} s")
    print("\n=== Per-approach timings (s) ===")
    for k in [x for x in ["NB","SB","EB","WB"] if x in plan.get("approaches", {})]:
        appr = plan["approaches"][k]
        print(f"{k}: green={appr['green']:.2f}, amber={appr['amber']:.2f}, red={appr['red']:.2f}")