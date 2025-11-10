"""
All-in-One Visualization Generator Script

This script runs the complete pipeline and generates all visualizations:
1. ML-based signal optimization (final.py logic)
2. Webster-based signal optimization (webster.py logic)
3. Performance comparison (compare_signal_plans.py logic)
4. SUMO simulation (sumo_simulation.py logic)
5. All visualization exports (PNG images + CSV dataset)

Usage: python generate_visualizations.py
"""

import json
import os
import sys
import subprocess
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import numpy as np
import random

# ============================================================================
# HARDCODED PCU VALUES (Fallback if intersection_summary.json doesn't exist)
# ============================================================================
# Uncomment the one you want to use:
PCU_FALLBACK = {"N": 2542.0, "S": 2760.0, "E": 0.0, "W": 1500.0}  # 3-way T-junction
# PCU_FALLBACK = {"N": 2880.0, "S": 2760.0, "E": 1560.0, "W": 3480.0}  # 4-way intersection

# Create outputs directory if it doesn't exist
OUTPUT_DIR = Path("outputs/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

print("=" * 80)
print("ALL-IN-ONE PIPELINE: Signal Optimization & Visualization Generator")
print("=" * 80)

# ============================================================================
# STEP 1: Run ML-based optimization (final.py)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: Running ML-based Signal Optimization")
print("=" * 80)

try:
    result = subprocess.run([sys.executable, "final.py"], 
                          capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        print("✅ ML optimization completed successfully")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    else:
        print("⚠️ ML optimization encountered an issue:")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
except Exception as e:
    print(f"⚠️ Could not run final.py: {e}")
    print("Continuing with available data...")

# ============================================================================
# STEP 2: Run Webster-based optimization (webster.py)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Running Webster-based Signal Optimization")
print("=" * 80)

try:
    result = subprocess.run([sys.executable, "webster.py"], 
                          capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        print("✅ Webster optimization completed successfully")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    else:
        print("⚠️ Webster optimization encountered an issue:")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
except Exception as e:
    print(f"⚠️ Could not run webster.py: {e}")
    print("Continuing with available data...")

# ============================================================================
# STEP 3: Run comparison (compare_signal_plans.py)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Comparing Signal Plans")
print("=" * 80)

try:
    result = subprocess.run([sys.executable, "compare_signal_plans.py"], 
                          capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        print("✅ Comparison completed successfully")
        # Print comparison summary
        lines = result.stdout.split('\n')
        summary_start = -1
        for i, line in enumerate(lines):
            if "OVERALL SUMMARY" in line:
                summary_start = i
                break
        if summary_start >= 0:
            print('\n'.join(lines[summary_start:]))
    else:
        print("⚠️ Comparison encountered an issue:")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
except Exception as e:
    print(f"⚠️ Could not run compare_signal_plans.py: {e}")
    print("Continuing with available data...")

# ============================================================================
# STEP 4: Run SUMO simulation (sumo_simulation.py)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Running SUMO Simulation")
print("=" * 80)

try:
    result = subprocess.run([sys.executable, "sumo_simulation.py"], 
                          capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        print("✅ SUMO simulation completed successfully")
        # Print key results
        lines = result.stdout.split('\n')
        for line in lines[-30:]:
            if line.strip():
                print(line)
    else:
        print("⚠️ SUMO simulation encountered an issue:")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
except Exception as e:
    print(f"⚠️ Could not run sumo_simulation.py: {e}")
    print("Continuing with visualization generation...")

# ============================================================================
# STEP 5: Generate all visualizations
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Generating All Visualizations")
print("=" * 80)
print(f"Output directory: {OUTPUT_DIR}\n")

def save_phase_diagram(signal_plan_path: str, method_name: str):
    """
    Generate and save phase diagram from signal plan JSON
    """
    print(f"Generating phase diagram for {method_name}...")
    
    if not os.path.exists(signal_plan_path):
        print(f"  ⚠️ {signal_plan_path} not found. Skipping.")
        return None
    
    with open(signal_plan_path, 'r') as f:
        plan = json.load(f)
    
    C = plan['cycle_length']
    approaches = plan['approaches']
    
    # Extract green times
    g_NS = 0
    g_EW = 0
    for approach_name, data in approaches.items():
        if approach_name in ['N', 'S', 'Northbound', 'Southbound']:
            g_NS += data.get('effective_green', 0)
        elif approach_name in ['E', 'W', 'Eastbound', 'Westbound']:
            g_EW += data.get('effective_green', 0)
    
    YELLOW = 3.0
    ALL_RED = 2.0
    
    colors = {"green": "#2ecc71", "amber": "#f1c40f", "red": "#e74c3c"}
    
    # Build phases
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
                                     marker_color=colors[kind], name=kind,
                                     showlegend=show, base=t))
            if g_EW > 0:
                fig.add_trace(go.Bar(x=[dur], y=["EW"], orientation="h",
                                     marker_color=colors[kind], name=kind,
                                     showlegend=False, base=t))
        t += dur
    
    fig.update_layout(
        barmode='stack',
        title=f"{method_name} Signal Timing - Phase Diagram (C={C:.1f}s)",
        xaxis_title='Time (seconds)',
        yaxis_title='Phase Group',
        height=400,
        width=1000,
        font=dict(size=14)
    )
    
    output_file = OUTPUT_DIR / f"phase_diagram_{method_name.lower().replace(' ', '_')}.png"
    fig.write_image(str(output_file), width=1000, height=400, scale=2)
    print(f"  ✅ Saved: {output_file}")
    
    return fig

def generate_comparison_charts():
    """
    Generate comparison bar charts for ML vs Webster
    """
    print("\nGenerating comparison charts...")
    
    # Load comparison data
    comparison_file = 'outputs/signalplan_comparision.json'
    if not os.path.exists(comparison_file):
        print(f"  ⚠️ {comparison_file} not found. Skipping comparison charts.")
        return
    
    with open(comparison_file, 'r') as f:
        data = json.load(f)
    
    ml_results = data.get('ml_plan', {})
    webster_results = data.get('webster_plan', {})
    
    if not ml_results or not webster_results:
        print("  ⚠️ No comparison data available.")
        return
    
    # Prepare data for charts
    approaches = list(ml_results.keys())
    metrics = ['delay', 'queue_length', 'throughput']
    
    for metric in metrics:
        # Filter approaches with valid values
        valid_approaches = [app for app in approaches 
                           if ml_results[app][metric] != float('inf') and 
                              webster_results[app][metric] != float('inf')]
        
        if not valid_approaches:
            continue
        
        ml_vals = [ml_results[app][metric] for app in valid_approaches]
        webster_vals = [webster_results[app][metric] for app in valid_approaches]
        
        # Create comparison bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='ML-based',
            x=valid_approaches,
            y=ml_vals,
            marker_color='#3498db'
        ))
        fig.add_trace(go.Bar(
            name='Webster-based',
            x=valid_approaches,
            y=webster_vals,
            marker_color='#e74c3c'
        ))
        
        fig.update_layout(
            title=f'{metric.replace("_", " ").title()} Comparison: ML vs Webster',
            xaxis_title='Approach',
            yaxis_title=metric.replace('_', ' ').title(),
            barmode='group',
            height=500,
            width=900,
            font=dict(size=12)
        )
        
        output_file = OUTPUT_DIR / f"{metric}_comparison.png"
        fig.write_image(str(output_file), width=900, height=500, scale=2)
        print(f"  ✅ Saved: {output_file}")
    
    # Average metrics comparison
    avg_metrics = {}
    for metric in metrics:
        ml_values = [ml_results[app][metric] for app in approaches 
                     if ml_results[app][metric] != float('inf')]
        webster_values = [webster_results[app][metric] for app in approaches 
                          if webster_results[app][metric] != float('inf')]
        
        avg_metrics[metric] = {
            'ML': np.mean(ml_values) if ml_values else 0,
            'Webster': np.mean(webster_values) if webster_values else 0
        }
    
    # Create overall comparison chart
    fig = go.Figure()
    
    metric_names = [m.replace('_', ' ').title() for m in metrics]
    ml_avgs = [avg_metrics[m]['ML'] for m in metrics]
    webster_avgs = [avg_metrics[m]['Webster'] for m in metrics]
    
    fig.add_trace(go.Bar(
        name='ML-based',
        x=metric_names,
        y=ml_avgs,
        marker_color='#3498db'
    ))
    fig.add_trace(go.Bar(
        name='Webster-based',
        x=metric_names,
        y=webster_avgs,
        marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        title='Average Performance Metrics: ML vs Webster',
        xaxis_title='Metric',
        yaxis_title='Average Value',
        barmode='group',
        height=500,
        width=900,
        font=dict(size=12)
    )
    
    output_file = OUTPUT_DIR / "average_metrics_comparison.png"
    fig.write_image(str(output_file), width=900, height=500, scale=2)
    print(f"  ✅ Saved: {output_file}")

def get_time_of_day_factor(hour):
    """Calculate time-of-day traffic factor"""
    if 7 <= hour < 10 or 17 <= hour < 20:
        return 1.4  # Peak hours
    elif 10 <= hour < 17:
        return 1.1  # Daytime
    elif 22 <= hour or hour < 6:
        return 0.5  # Night
    else:
        return 1.0  # Normal

def get_weather_factor(weather):
    """Calculate weather impact factor"""
    weather_map = {"clear": 1.0, "rain": 1.2, "fog": 1.15}
    return weather_map.get(weather, 1.0)

def get_event_factor(has_event):
    """Calculate special event factor"""
    return 1.3 if has_event else 1.0

def get_day_of_week_factor(day):
    """Calculate day of week factor"""
    return 0.8 if day in ['Saturday', 'Sunday'] else 1.0

def export_synthetic_dataset():
    """
    Generate and export synthetic training dataset to CSV
    """
    print("\nGenerating synthetic dataset...")
    
    # Generate synthetic data
    num_samples = 1000
    data = []
    
    for i in range(num_samples):
        # Base PCU values
        N = random.randint(100, 4000)
        S = random.randint(100, 4000)
        E = random.randint(100, 4000)
        W = random.randint(100, 4000)
        
        # Context
        hour = random.randint(0, 23)
        weather = random.choice(["clear", "clear", "clear", "rain", "fog"])
        has_event = random.random() < 0.1
        day = random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        # Calculate factors
        tod_factor = get_time_of_day_factor(hour)
        weather_factor = get_weather_factor(weather)
        event_factor = get_event_factor(has_event)
        day_factor = get_day_of_week_factor(day)
        
        # Apply skew
        combined_factor = tod_factor * weather_factor * event_factor * day_factor
        N_real = int(N * combined_factor)
        S_real = int(S * combined_factor)
        E_real = int(E * combined_factor)
        W_real = int(W * combined_factor)
        
        # Calculate Webster cycle and delay
        Y_total = (N_real + S_real + E_real + W_real) / 1800.0
        L = 12.0
        C_opt = (1.5 * L + 5) / (1 - Y_total) if Y_total < 1 else 120
        C_opt = max(30, min(120, C_opt))
        
        # Calculate delay (simplified)
        delay = C_opt * 0.3 + random.uniform(-5, 5)
        
        data.append({
            'N': N,
            'S': S,
            'E': E,
            'W': W,
            'N_real': N_real,
            'S_real': S_real,
            'E_real': E_real,
            'W_real': W_real,
            'hour': hour,
            'weather': weather,
            'has_event': has_event,
            'day_of_week': day,
            'time_of_day_factor': tod_factor,
            'weather_factor': weather_factor,
            'event_factor': event_factor,
            'day_factor': day_factor,
            'combined_factor': combined_factor,
            'cycle_length': C_opt,
            'delay': delay
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    output_file = OUTPUT_DIR / "synthetic_training_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"  ✅ Saved dataset with {len(df)} samples: {output_file}")
    
    # Generate dataset distribution plots
    print("\nGenerating dataset distribution plots...")
    
    # PCU distribution
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['N_real'], name='North', opacity=0.7))
    fig.add_trace(go.Histogram(x=df['S_real'], name='South', opacity=0.7))
    fig.add_trace(go.Histogram(x=df['E_real'], name='East', opacity=0.7))
    fig.add_trace(go.Histogram(x=df['W_real'], name='West', opacity=0.7))
    fig.update_layout(
        title='PCU Distribution Across Approaches',
        xaxis_title='PCU Value',
        yaxis_title='Frequency',
        barmode='overlay',
        height=500,
        width=900
    )
    output_file = OUTPUT_DIR / "pcu_distribution.png"
    fig.write_image(str(output_file), width=900, height=500, scale=2)
    print(f"  ✅ Saved: {output_file}")
    
    # Time of day distribution
    fig = px.histogram(df, x='hour', color='weather', 
                       title='Traffic Distribution by Hour of Day',
                       labels={'hour': 'Hour of Day', 'count': 'Frequency'})
    fig.update_layout(height=500, width=900)
    output_file = OUTPUT_DIR / "time_of_day_distribution.png"
    fig.write_image(str(output_file), width=900, height=500, scale=2)
    print(f"  ✅ Saved: {output_file}")
    
    # Cycle length distribution
    fig = px.histogram(df, x='cycle_length', nbins=30,
                       title='Optimized Cycle Length Distribution',
                       labels={'cycle_length': 'Cycle Length (seconds)', 'count': 'Frequency'})
    fig.update_layout(height=500, width=900)
    output_file = OUTPUT_DIR / "cycle_length_distribution.png"
    fig.write_image(str(output_file), width=900, height=500, scale=2)
    print(f"  ✅ Saved: {output_file}")
    
    # Weather impact
    weather_avg = df.groupby('weather')[['N_real', 'S_real', 'E_real', 'W_real']].mean()
    fig = go.Figure()
    for col in ['N_real', 'S_real', 'E_real', 'W_real']:
        fig.add_trace(go.Bar(name=col.replace('_real', ''), x=weather_avg.index, y=weather_avg[col]))
    fig.update_layout(
        title='Average PCU by Weather Condition',
        xaxis_title='Weather',
        yaxis_title='Average PCU',
        barmode='group',
        height=500,
        width=900
    )
    output_file = OUTPUT_DIR / "weather_impact.png"
    fig.write_image(str(output_file), width=900, height=500, scale=2)
    print(f"  ✅ Saved: {output_file}")
    
    # Day of week impact
    day_avg = df.groupby('day_of_week')[['N_real', 'S_real', 'E_real', 'W_real']].mean()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_avg = day_avg.reindex(day_order)
    fig = go.Figure()
    for col in ['N_real', 'S_real', 'E_real', 'W_real']:
        fig.add_trace(go.Bar(name=col.replace('_real', ''), x=day_avg.index, y=day_avg[col]))
    fig.update_layout(
        title='Average PCU by Day of Week',
        xaxis_title='Day',
        yaxis_title='Average PCU',
        barmode='group',
        height=500,
        width=900
    )
    output_file = OUTPUT_DIR / "day_of_week_impact.png"
    fig.write_image(str(output_file), width=900, height=500, scale=2)
    print(f"  ✅ Saved: {output_file}")

def generate_sumo_comparison_charts():
    """
    Generate SUMO performance comparison charts
    """
    print("\nGenerating SUMO comparison charts...")
    
    sumo_file = 'outputs/sumo_comparison.json'
    if not os.path.exists(sumo_file):
        print(f"  ⚠️ {sumo_file} not found. Skipping SUMO charts.")
        return
    
    with open(sumo_file, 'r') as f:
        data = json.load(f)
    
    ml_metrics = data.get('ML', {})
    webster_metrics = data.get('Webster', {})
    
    if not ml_metrics or not webster_metrics:
        print("  ⚠️ No SUMO comparison data available.")
        return
    
    metrics = ['average_delay', 'average_waiting_time', 'average_travel_time', 
               'average_time_loss', 'total_throughput']
    metric_labels = ['Avg Delay (s)', 'Avg Waiting Time (s)', 'Avg Travel Time (s)', 
                     'Avg Time Loss (s)', 'Total Throughput (vehicles)']
    
    # Individual metric comparisons
    for metric, label in zip(metrics, metric_labels):
        if metric not in ml_metrics or metric not in webster_metrics:
            continue
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Method',
            x=['ML-based', 'Webster-based'],
            y=[ml_metrics[metric], webster_metrics[metric]],
            marker_color=['#3498db', '#e74c3c']
        ))
        
        fig.update_layout(
            title=f'SUMO Simulation: {label}',
            yaxis_title=label,
            height=500,
            width=700,
            showlegend=False
        )
        
        output_file = OUTPUT_DIR / f"sumo_{metric}.png"
        fig.write_image(str(output_file), width=700, height=500, scale=2)
        print(f"  ✅ Saved: {output_file}")
    
    # All metrics comparison (normalized)
    fig = go.Figure()
    
    # Normalize throughput separately (it's the only metric where higher is better)
    delay_metrics = metrics[:-1]  # All except throughput
    delay_labels = metric_labels[:-1]
    
    ml_delays = [ml_metrics.get(m, 0) for m in delay_metrics]
    webster_delays = [webster_metrics.get(m, 0) for m in delay_metrics]
    
    fig.add_trace(go.Bar(
        name='ML-based',
        x=delay_labels,
        y=ml_delays,
        marker_color='#3498db'
    ))
    fig.add_trace(go.Bar(
        name='Webster-based',
        x=delay_labels,
        y=webster_delays,
        marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        title='SUMO Simulation: Time-based Performance Metrics (Lower is Better)',
        xaxis_title='Metric',
        yaxis_title='Time (seconds)',
        barmode='group',
        height=600,
        width=1000,
        font=dict(size=11)
    )
    
    output_file = OUTPUT_DIR / "sumo_time_metrics_comparison.png"
    fig.write_image(str(output_file), width=1000, height=600, scale=2)
    print(f"  ✅ Saved: {output_file}")
    
    # Throughput comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Method',
        x=['ML-based', 'Webster-based'],
        y=[ml_metrics.get('total_throughput', 0), webster_metrics.get('total_throughput', 0)],
        marker_color=['#3498db', '#e74c3c']
    ))
    
    fig.update_layout(
        title='SUMO Simulation: Throughput (Higher is Better)',
        yaxis_title='Total Vehicles',
        height=500,
        width=700,
        showlegend=False
    )
    
    output_file = OUTPUT_DIR / "sumo_throughput_comparison.png"
    fig.write_image(str(output_file), width=700, height=500, scale=2)
    print(f"  ✅ Saved: {output_file}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# 1. Phase diagrams
if os.path.exists('outputs/ml_signal_plan.json'):
    save_phase_diagram('outputs/ml_signal_plan.json', 'ML')
else:
    print("⚠️ outputs/ml_signal_plan.json not found.")

if os.path.exists('outputs/webster_signal_plan.json'):
    save_phase_diagram('outputs/webster_signal_plan.json', 'Webster')
else:
    print("⚠️ outputs/webster_signal_plan.json not found.")

# 2. Comparison charts
generate_comparison_charts()

# 3. Synthetic dataset
export_synthetic_dataset()

# 4. SUMO comparison
generate_sumo_comparison_charts()

print("\n" + "=" * 80)
print("✅ COMPLETE PIPELINE FINISHED!")
print("=" * 80)
print(f"\n📁 All visualizations saved to: {OUTPUT_DIR}")

# List all generated files
if OUTPUT_DIR.exists():
    files = sorted(OUTPUT_DIR.glob('*'))
    if files:
        print(f"\n📊 Generated {len(files)} files:")
        for f in files:
            size = f.stat().st_size / 1024  # KB
            print(f"  - {f.name} ({size:.1f} KB)")
    
    # Summary
    png_count = len(list(OUTPUT_DIR.glob('*.png')))
    csv_count = len(list(OUTPUT_DIR.glob('*.csv')))
    print(f"\n✅ Summary: {png_count} PNG images + {csv_count} CSV dataset")
    print(f"📂 Copy these to report/images/ for your LaTeX report")

print("\n" + "=" * 80)
print("Next steps:")
print("1. Copy images from outputs/visualizations/ to report/images/")
print("2. Capture Streamlit UI screenshots (see VISUALIZATION_GUIDE.md)")
print("3. Capture SUMO GUI screenshots (see VISUALIZATION_GUIDE.md)")
print("4. Compile LaTeX report (cd report && pdflatex main.tex)")
print("=" * 80)
