"""
Complete pipeline to generate ML vs Webster comparison for 2 intersections.

Workflow:
1. Generate training CSV once
2. Train ML models once
3. For each intersection:
   - Run ML optimization
   - Run Webster optimization
   - Run SUMO simulation for ML
   - Run SUMO simulation for Webster
   - Generate comparison charts
   - Save images to report/images/intersection {n}/
"""

import json
import os
import sys
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Import modules
import final
import webster
import sumo_simulation

# For saving Plotly figures as images
try:
    import plotly.graph_objects as go
    # Check if kaleido is available (needed for write_image)
    try:
        import kaleido
    except ImportError:
        pass  # kaleido might be installed but not importable directly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Phase diagrams will not be saved.")

# PCU data for 2 intersections
INTERSECTION_1_PCU = {"N": 2542.0, "S": 2760.0, "E": 0.0, "W": 1500.0}  # 3-way
INTERSECTION_2_PCU = {"N": 2880.0, "S": 2760.0, "E": 1560.0, "W": 3480.0}  # 4-way


def generate_comparison_charts(ml_metrics: dict, webster_metrics: dict, output_dir: str):
    """
    Generate 8 comparison charts showing ML vs Webster performance.
    
    Args:
        ml_metrics: Dictionary with SUMO metrics from ML simulation
        webster_metrics: Dictionary with SUMO metrics from Webster simulation
        output_dir: Directory to save images (e.g., 'report/images/intersection 1')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    ml_delay = ml_metrics.get('avg_delay', 0)
    ml_waiting = ml_metrics.get('avg_waiting_time', 0)
    ml_travel = ml_metrics.get('avg_travel_time', 0)
    ml_time_loss = ml_metrics.get('avg_time_loss', 0)
    ml_throughput = ml_metrics.get('vehicle_count', 0)
    
    webster_delay = webster_metrics.get('avg_delay', 0)
    webster_waiting = webster_metrics.get('avg_waiting_time', 0)
    webster_travel = webster_metrics.get('avg_travel_time', 0)
    webster_time_loss = webster_metrics.get('avg_time_loss', 0)
    webster_throughput = webster_metrics.get('vehicle_count', 0)
    
    # Calculate improvement percentage
    def calc_improvement(ml_val, webster_val):
        if webster_val == 0:
            return 0.0
        return ((webster_val - ml_val) / webster_val) * 100
    
    delay_improvement = calc_improvement(ml_delay, webster_delay)
    waiting_improvement = calc_improvement(ml_waiting, webster_waiting)
    travel_improvement = calc_improvement(ml_travel, webster_travel)
    time_loss_improvement = calc_improvement(ml_time_loss, webster_time_loss)
    
    # Determine winner for delay chart
    delay_winner = "ML-based" if ml_delay < webster_delay else "Webster-based"
    delay_diff = abs(ml_delay - webster_delay)
    delay_pct = (delay_diff / max(webster_delay, 0.01)) * 100
    
    # 1. Average Delay Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = ['ML-based', 'Webster-based']
    delays = [ml_delay, webster_delay]
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(methods, delays, color=colors, alpha=0.8)
    ax.set_ylabel('Average Delay (s)', fontsize=12)
    ax.set_title(f'SUMO Simulation: Average Delay (s)\nWinner: {delay_winner} (Δ {delay_diff:.2f}, {delay_pct:.1f}%)', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, delay in zip(bars, delays):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{delay:.2f}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sumo_average_delay.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Average Waiting Time
    fig, ax = plt.subplots(figsize=(8, 6))
    waiting_times = [ml_waiting, webster_waiting]
    bars = ax.bar(methods, waiting_times, color=colors, alpha=0.8)
    ax.set_ylabel('Average Waiting Time (s)', fontsize=12)
    ax.set_title('SUMO Simulation: Average Waiting Time (s)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, wt in zip(bars, waiting_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{wt:.2f}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sumo_average_waiting_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Average Travel Time
    fig, ax = plt.subplots(figsize=(8, 6))
    travel_times = [ml_travel, webster_travel]
    bars = ax.bar(methods, travel_times, color=colors, alpha=0.8)
    ax.set_ylabel('Average Travel Time (s)', fontsize=12)
    ax.set_title('SUMO Simulation: Average Travel Time (s)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, tt in zip(bars, travel_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{tt:.2f}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sumo_average_travel_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Average Time Loss
    fig, ax = plt.subplots(figsize=(8, 6))
    time_losses = [ml_time_loss, webster_time_loss]
    bars = ax.bar(methods, time_losses, color=colors, alpha=0.8)
    ax.set_ylabel('Average Time Loss (s)', fontsize=12)
    ax.set_title('SUMO Simulation: Average Time Loss (s)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, tl in zip(bars, time_losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{tl:.2f}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sumo_average_time_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Throughput Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    throughputs = [ml_throughput, webster_throughput]
    bars = ax.bar(methods, throughputs, color=colors, alpha=0.8)
    ax.set_ylabel('Total Throughput (vehicles)', fontsize=12)
    ax.set_title('SUMO Simulation: Total Throughput', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, tp in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughputs)*0.01,
                f'{int(tp)}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sumo_total_throughput.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Throughput Comparison (alternative)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(methods, throughputs, color=colors, alpha=0.8)
    ax.set_ylabel('Vehicle Throughput', fontsize=12)
    ax.set_title('SUMO Simulation: Throughput Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, tp in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughputs)*0.01,
                f'{int(tp)}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sumo_throughput_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Time Metrics Comparison (all together)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.35
    metrics_labels = ['Delay', 'Waiting', 'Travel', 'Time Loss']
    ml_values = [ml_delay, ml_waiting, ml_travel, ml_time_loss]
    webster_values = [webster_delay, webster_waiting, webster_travel, webster_time_loss]
    
    x_pos = np.arange(len(metrics_labels))
    ax.bar(x_pos - width/2, ml_values, width, label='ML-based', color='#3498db', alpha=0.8)
    ax.bar(x_pos + width/2, webster_values, width, label='Webster-based', color='#e74c3c', alpha=0.8)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_title('SUMO Simulation: Time Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sumo_time_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. ML Improvement Percentage
    fig, ax = plt.subplots(figsize=(8, 6))
    improvements = [delay_improvement, waiting_improvement, travel_improvement, time_loss_improvement]
    metric_names = ['Delay', 'Waiting\nTime', 'Travel\nTime', 'Time\nLoss']
    colors_bar = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax.bar(metric_names, improvements, color=colors_bar, alpha=0.8)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('ML-based vs Webster: Performance Improvement', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if imp > 0 else -2),
                f'{imp:.1f}%', ha='center', va='bottom' if imp > 0 else 'top', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sumo_ml_improvement_percentage.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Generated 8 comparison charts in {output_dir}")


def generate_combined_comparison_charts(int1_ml: dict, int1_webster: dict, 
                                       int2_ml: dict, int2_webster: dict, 
                                       output_dir: str):
    """
    Generate 2 combined comparison charts showing both intersections:
    1. Time-based metrics comparison (delay, waiting, travel, time loss)
    2. Traffic-based metrics comparison (throughput)
    
    Args:
        int1_ml: Intersection 1 ML metrics
        int1_webster: Intersection 1 Webster metrics
        int2_ml: Intersection 2 ML metrics
        int2_webster: Intersection 2 Webster metrics
        output_dir: Directory to save images (e.g., 'report/images')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics for Intersection 1
    int1_ml_delay = int1_ml.get('avg_delay', 0)
    int1_ml_waiting = int1_ml.get('avg_waiting_time', 0)
    int1_ml_travel = int1_ml.get('avg_travel_time', 0)
    int1_ml_time_loss = int1_ml.get('avg_time_loss', 0)
    int1_ml_throughput = int1_ml.get('vehicle_count', 0)
    
    int1_webster_delay = int1_webster.get('avg_delay', 0)
    int1_webster_waiting = int1_webster.get('avg_waiting_time', 0)
    int1_webster_travel = int1_webster.get('avg_travel_time', 0)
    int1_webster_time_loss = int1_webster.get('avg_time_loss', 0)
    int1_webster_throughput = int1_webster.get('vehicle_count', 0)
    
    # Extract metrics for Intersection 2
    int2_ml_delay = int2_ml.get('avg_delay', 0)
    int2_ml_waiting = int2_ml.get('avg_waiting_time', 0)
    int2_ml_travel = int2_ml.get('avg_travel_time', 0)
    int2_ml_time_loss = int2_ml.get('avg_time_loss', 0)
    int2_ml_throughput = int2_ml.get('vehicle_count', 0)
    
    int2_webster_delay = int2_webster.get('avg_delay', 0)
    int2_webster_waiting = int2_webster.get('avg_waiting_time', 0)
    int2_webster_travel = int2_webster.get('avg_travel_time', 0)
    int2_webster_time_loss = int2_webster.get('avg_time_loss', 0)
    int2_webster_throughput = int2_webster.get('vehicle_count', 0)
    
    # 1. Time-based metrics comparison (both intersections)
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(4)  # 4 metrics: delay, waiting, travel, time_loss
    width = 0.2
    
    # Intersection 1 data
    int1_ml_values = [int1_ml_delay, int1_ml_waiting, int1_ml_travel, int1_ml_time_loss]
    int1_webster_values = [int1_webster_delay, int1_webster_waiting, int1_webster_travel, int1_webster_time_loss]
    
    # Intersection 2 data
    int2_ml_values = [int2_ml_delay, int2_ml_waiting, int2_ml_travel, int2_ml_time_loss]
    int2_webster_values = [int2_webster_delay, int2_webster_waiting, int2_webster_travel, int2_webster_time_loss]
    
    # Plot bars with intersection names
    ax.bar(x - 1.5*width, int1_ml_values, width, label='Jyoti Circle (T-junction): ML', color='#3498db', alpha=0.8)
    ax.bar(x - 0.5*width, int1_webster_values, width, label='Jyoti Circle (T-junction): Webster', color='#2ecc71', alpha=0.8)
    ax.bar(x + 0.5*width, int2_ml_values, width, label='Hampankatta Circle (4-approach): ML', color='#e74c3c', alpha=0.8)
    ax.bar(x + 1.5*width, int2_webster_values, width, label='Hampankatta Circle (4-approach): Webster', color='#f39c12', alpha=0.8)
    
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time-based Metrics', fontsize=12, fontweight='bold')
    ax.set_title('Time-based Performance Comparison: ML vs Webster\nAcross Both Intersections', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Average\nDelay', 'Average\nWaiting Time', 'Average\nTravel Time', 'Average\nTime Loss'])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (v1, v2, v3, v4) in enumerate(zip(int1_ml_values, int1_webster_values, int2_ml_values, int2_webster_values)):
        ax.text(i - 1.5*width, v1 + 1, f'{v1:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(i - 0.5*width, v2 + 1, f'{v2:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + 0.5*width, v3 + 1, f'{v3:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + 1.5*width, v4 + 1, f'{v4:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_time_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Traffic-based metrics comparison (throughput)
    fig, ax = plt.subplots(figsize=(10, 6))
    intersections = ['Jyoti Circle\n(T-junction)', 'Hampankatta Circle\n(4-approach)']
    x = np.arange(len(intersections))
    width = 0.35
    
    ml_throughputs = [int1_ml_throughput, int2_ml_throughput]
    webster_throughputs = [int1_webster_throughput, int2_webster_throughput]
    
    bars1 = ax.bar(x - width/2, ml_throughputs, width, label='ML-based', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, webster_throughputs, width, label='Webster-based', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Total Throughput (vehicles)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Intersection', fontsize=12, fontweight='bold')
    ax.set_title('Traffic Throughput Comparison: ML vs Webster\nAcross Both Intersections', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(intersections)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(max(ml_throughputs), max(webster_throughputs))*0.01,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_traffic_throughput_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Generated 2 combined comparison charts in {output_dir}")


def run_ml_sumo_pipeline(pcu_dict: dict, intersection_num: int, ml_plan_path: str):
    """
    Run SUMO simulation for ML signal plan.
    
    Returns metrics dictionary.
    """
    intersection_dir = f'outputs/intersection_{intersection_num}'
    os.makedirs(intersection_dir, exist_ok=True)
    
    # Load ML plan
    with open(ml_plan_path, 'r') as f:
        ml_plan = json.load(f)
    
    # Create intersection summary for routes (convert N/S/E/W to NB/SB/EB/WB)
    intersection_data = {}
    approach_map = {"N": "NB", "S": "SB", "E": "EB", "W": "WB"}
    for appr_code, appr_name in approach_map.items():
        pcu_val = pcu_dict.get(appr_code, 0.0)
        if pcu_val > 0:
            intersection_data[appr_name] = {'total_pcu': pcu_val}
    
    # Create SUMO files with unique names for this intersection (use os.path.join for consistency)
    net_file = os.path.join(intersection_dir, 'sumo_network.net.xml')
    add_file = os.path.join(intersection_dir, 'ml_traffic_lights.add.xml')
    route_file = os.path.join(intersection_dir, 'routes.rou.xml')
    config_file = os.path.join(intersection_dir, 'ml_config.sumocfg')
    tripinfo_file = os.path.join(intersection_dir, 'tripinfo_ML.xml')
    metrics_file = os.path.join(intersection_dir, 'sumo_metrics_ml.json')
    
    # Detect present approaches
    present_approaches = set()
    for appr_id, data in ml_plan.get('approaches', {}).items():
        if data.get('green', 0) > 0:
            present_approaches.add(appr_id)
    if not present_approaches:
        present_approaches = {'NB', 'SB', 'EB', 'WB'}
    
    # Create SUMO network
    sumo_simulation.create_sumo_network([ml_plan], net_file)
    
    # Create traffic lights
    sumo_simulation.create_traffic_lights(ml_plan, add_file, 'ML', net_file)
    
    # Create routes (use same intersection_data format)
    sumo_simulation.create_routes(
        intersection_data,
        present_approaches,
        route_file,
        sumo_simulation.SIMULATION_TIME,
        None,
        net_file
    )
    
    # Create SUMO config
    sumo_simulation.create_sumo_config(net_file, route_file, add_file, config_file, 'ML')
    
    # Run SUMO simulation
    metrics = None
    if sumo_simulation.SUMO_AVAILABLE:
        success = sumo_simulation.run_sumo_simulation(config_file, 'ML', use_gui=False)
        if success:
            # Tripinfo file is created in the config directory (same as intersection_dir)
            # since SUMO runs from config directory
            tripinfo_actual = os.path.normpath(os.path.join(intersection_dir, 'tripinfo_ML.xml'))
            tripinfo_file_norm = os.path.normpath(tripinfo_file)
            
            if os.path.exists(tripinfo_actual):
                # Check if paths are actually different (normalized)
                if tripinfo_actual != tripinfo_file_norm:
                    # Move to desired location
                    if os.path.exists(tripinfo_file_norm):
                        os.remove(tripinfo_file_norm)
                    shutil.move(tripinfo_actual, tripinfo_file_norm)
                # If paths are the same, file is already in the right place
                metrics = sumo_simulation.extract_sumo_metrics(tripinfo_file_norm)
            elif os.path.exists(tripinfo_file_norm):
                # File already in the right place
                metrics = sumo_simulation.extract_sumo_metrics(tripinfo_file_norm)
            else:
                print(f"  Warning: Tripinfo file not found at {tripinfo_actual} or {tripinfo_file_norm}")
                return None
            if metrics:
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
        else:
            print(f"  Warning: SUMO simulation failed for ML plan")
    else:
        print(f"  Warning: SUMO not available, skipping simulation")
    
    return metrics


def run_intersection_pipeline(intersection_num: int, pcu_dict: dict, 
                              cycle_model, green_model, cycle_features, green_features):
    """
    Run complete pipeline for one intersection:
    1. ML optimization
    2. Webster optimization
    3. SUMO simulation for ML
    4. SUMO simulation for Webster
    5. Generate comparison charts
    """
    print(f"\n{'='*80}")
    print(f"INTERSECTION {intersection_num}")
    print(f"{'='*80}")
    print(f"PCU: {pcu_dict}")
    
    intersection_dir = os.path.join('outputs', f'intersection_{intersection_num}')
    image_dir = os.path.join('report', 'images', f'intersection {intersection_num}')
    os.makedirs(intersection_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # 1. ML Optimization
    print(f"\n[1/5] Running ML optimization...")
    ml_plan_path = os.path.join(intersection_dir, 'ml_signal_plan.json')
    ml_plan = final.predict_ml_signal_plan(
        pcu_dict, cycle_model, green_model, cycle_features, green_features, ml_plan_path
    )
    print(f"  ML cycle length: {ml_plan['cycle_length']:.2f}s")
    
    # 2. Webster Optimization
    print(f"\n[2/5] Running Webster optimization...")
    webster_plan_path = os.path.join(intersection_dir, 'webster_signal_plan.json')
    webster_plan = webster.compute_webster_plan(
        pcu_override=pcu_dict,
        output_path=webster_plan_path,
        save_json=True,
        ensure_outputs_dir=True
    )
    print(f"  Webster cycle length: {webster_plan['cycle_length']:.2f}s")
    
    # 3. SUMO Simulation for ML
    print(f"\n[3/5] Running SUMO simulation for ML plan...")
    ml_metrics = run_ml_sumo_pipeline(pcu_dict, intersection_num, ml_plan_path)
    if ml_metrics:
        print(f"\n  ML-BASED METHOD METRICS:")
        print(f"    Vehicle Count (Throughput):    {ml_metrics.get('vehicle_count', 0):,} vehicles")
        print(f"    Average Delay:                 {ml_metrics.get('avg_delay', 0):.2f} seconds")
        print(f"    Average Waiting Time:          {ml_metrics.get('avg_waiting_time', 0):.2f} seconds")
        print(f"    Average Travel Time:           {ml_metrics.get('avg_travel_time', 0):.2f} seconds")
        print(f"    Average Time Loss:            {ml_metrics.get('avg_time_loss', 0):.2f} seconds")
        print(f"    Average Departure Delay:       {ml_metrics.get('avg_depart_delay', 0):.2f} seconds")
        print(f"    Total Delay:                   {ml_metrics.get('total_delay', 0):,.0f} seconds")
        print(f"    Total Waiting Time:            {ml_metrics.get('total_waiting_time', 0):,.0f} seconds")
        print(f"\n  ML SIGNAL PLAN:")
        print(f"    Cycle Length:                 {ml_plan['cycle_length']:.2f} seconds")
        for appr_id, appr_data in ml_plan.get('approaches', {}).items():
            if appr_data.get('green', 0) > 0:
                print(f"    {appr_id}:")
                print(f"      Green Time:              {appr_data.get('green', 0):.2f} seconds")
                print(f"      Amber Time:             {appr_data.get('amber', 0):.2f} seconds")
                print(f"      Red Time:               {appr_data.get('red', 0):.2f} seconds")
                print(f"      Flow Rate:              {appr_data.get('arrival_flow_rate', 0):.0f} PCU/hr")
    
    # 4. SUMO Simulation for Webster
    print(f"\n[4/5] Running SUMO simulation for Webster plan...")
    webster_metrics_file = os.path.join(intersection_dir, 'sumo_metrics_webster.json')
    webster_results = sumo_simulation.run_webster_pipeline(
        pcu_override=pcu_dict,
        run_sumo=True,
        use_gui=False,
        plan_output=webster_plan_path,
        net_output=f'{intersection_dir}/sumo_network_webster.net.xml',
        add_output=f'{intersection_dir}/webster_traffic_lights.add.xml',
        route_output=f'{intersection_dir}/routes_webster.rou.xml',
        config_output=f'{intersection_dir}/webster_config.sumocfg',
        metrics_output=webster_metrics_file
    )
    webster_metrics = webster_results.get('metrics')
    if webster_metrics:
        print(f"\n  WEBSTER-BASED METHOD METRICS:")
        print(f"    Vehicle Count (Throughput):    {webster_metrics.get('vehicle_count', 0):,} vehicles")
        print(f"    Average Delay:                 {webster_metrics.get('avg_delay', 0):.2f} seconds")
        print(f"    Average Waiting Time:          {webster_metrics.get('avg_waiting_time', 0):.2f} seconds")
        print(f"    Average Travel Time:           {webster_metrics.get('avg_travel_time', 0):.2f} seconds")
        print(f"    Average Time Loss:             {webster_metrics.get('avg_time_loss', 0):.2f} seconds")
        print(f"    Average Departure Delay:       {webster_metrics.get('avg_depart_delay', 0):.2f} seconds")
        print(f"    Total Delay:                   {webster_metrics.get('total_delay', 0):,.0f} seconds")
        print(f"    Total Waiting Time:            {webster_metrics.get('total_waiting_time', 0):,.0f} seconds")
        print(f"\n  WEBSTER SIGNAL PLAN:")
        print(f"    Cycle Length:                 {webster_plan['cycle_length']:.2f} seconds")
        for appr_id, appr_data in webster_plan.get('approaches', {}).items():
            if appr_data.get('green', 0) > 0:
                print(f"    {appr_id}:")
                print(f"      Green Time:              {appr_data.get('green', 0):.2f} seconds")
                print(f"      Amber Time:             {appr_data.get('amber', 0):.2f} seconds")
                print(f"      Red Time:               {appr_data.get('red', 0):.2f} seconds")
                print(f"      Flow Rate:              {appr_data.get('arrival_flow_rate', 0):.0f} PCU/hr")
    
    # Print comparison
    if ml_metrics and webster_metrics:
        print(f"\n  COMPARISON:")
        ml_delay = ml_metrics.get('avg_delay', 0)
        webster_delay = webster_metrics.get('avg_delay', 0)
        delay_diff = ml_delay - webster_delay
        delay_pct = (delay_diff / webster_delay * 100) if webster_delay > 0 else 0
        delay_winner = "ML" if ml_delay < webster_delay else "Webster"
        print(f"    Average Delay:")
        print(f"      ML:      {ml_delay:.2f} seconds")
        print(f"      Webster: {webster_delay:.2f} seconds")
        print(f"      Difference: {delay_diff:+.2f} seconds ({delay_pct:+.1f}%)")
        print(f"      → Winner: {delay_winner}-based method")
        
        ml_travel = ml_metrics.get('avg_travel_time', 0)
        webster_travel = webster_metrics.get('avg_travel_time', 0)
        travel_diff = ml_travel - webster_travel
        travel_pct = (travel_diff / webster_travel * 100) if webster_travel > 0 else 0
        travel_winner = "ML" if ml_travel < webster_travel else "Webster"
        print(f"\n    Average Travel Time:")
        print(f"      ML:      {ml_travel:.2f} seconds")
        print(f"      Webster: {webster_travel:.2f} seconds")
        print(f"      Difference: {travel_diff:+.2f} seconds ({travel_pct:+.1f}%)")
        print(f"      → Winner: {travel_winner}-based method")
        
        ml_time_loss = ml_metrics.get('avg_time_loss', 0)
        webster_time_loss = webster_metrics.get('avg_time_loss', 0)
        time_loss_diff = ml_time_loss - webster_time_loss
        time_loss_pct = (time_loss_diff / webster_time_loss * 100) if webster_time_loss > 0 else 0
        time_loss_winner = "ML" if ml_time_loss < webster_time_loss else "Webster"
        print(f"\n    Average Time Loss:")
        print(f"      ML:      {ml_time_loss:.2f} seconds")
        print(f"      Webster: {webster_time_loss:.2f} seconds")
        print(f"      Difference: {time_loss_diff:+.2f} seconds ({time_loss_pct:+.1f}%)")
        print(f"      → Winner: {time_loss_winner}-based method")
        
        ml_throughput = ml_metrics.get('vehicle_count', 0)
        webster_throughput = webster_metrics.get('vehicle_count', 0)
        throughput_diff = ml_throughput - webster_throughput
        throughput_pct = (throughput_diff / webster_throughput * 100) if webster_throughput > 0 else 0
        throughput_winner = "ML" if ml_throughput > webster_throughput else "Webster"
        print(f"\n    Throughput:")
        print(f"      ML:      {ml_throughput:,} vehicles")
        print(f"      Webster: {webster_throughput:,} vehicles")
        print(f"      Difference: {throughput_diff:+,} vehicles ({throughput_pct:+.1f}%)")
        print(f"      → Winner: {throughput_winner}-based method (higher is better)")
        
        cycle_diff = ml_plan['cycle_length'] - webster_plan['cycle_length']
        cycle_winner = "ML" if abs(cycle_diff) < 1.0 else ("ML" if ml_plan['cycle_length'] < webster_plan['cycle_length'] else "Webster")
        print(f"\n    Cycle Length:")
        print(f"      ML:      {ml_plan['cycle_length']:.2f} seconds")
        print(f"      Webster: {webster_plan['cycle_length']:.2f} seconds")
        print(f"      Difference: {cycle_diff:+.2f} seconds")
        if abs(cycle_diff) < 1.0:
            print(f"      → Similar cycle lengths (difference < 1s)")
        else:
            print(f"      → Shorter cycle: {cycle_winner}-based method")
    
    # 5. Generate Comparison Charts
    print(f"\n[5/6] Generating comparison charts...")
    if ml_metrics and webster_metrics:
        generate_comparison_charts(ml_metrics, webster_metrics, image_dir)
        print(f"  Charts saved to {image_dir}")
    else:
        print(f"  Warning: Missing metrics, skipping chart generation")
        if not ml_metrics:
            print(f"    ML metrics missing")
        if not webster_metrics:
            print(f"    Webster metrics missing")
    
    # 6. Generate Phase Diagrams
    print(f"\n[6/6] Generating phase diagrams...")
    if PLOTLY_AVAILABLE:
        try:
            # ML Phase Diagram
            ml_phase_fig = webster.create_phase_diagram(ml_plan)
            ml_phase_fig.update_layout(title=f"ML Phase Diagram - Intersection {intersection_num} (Cycle={ml_plan['cycle_length']:.1f}s)")
            ml_phase_path = os.path.join(image_dir, f'phase_diagram_ml.png')
            ml_phase_fig.write_image(ml_phase_path, width=1200, height=400, scale=2)
            print(f"  ML phase diagram saved: {ml_phase_path}")
            
            # Webster Phase Diagram
            webster_phase_fig = webster.create_phase_diagram(webster_plan)
            webster_phase_fig.update_layout(title=f"Webster Phase Diagram - Intersection {intersection_num} (Cycle={webster_plan['cycle_length']:.1f}s)")
            webster_phase_path = os.path.join(image_dir, f'phase_diagram_webster.png')
            webster_phase_fig.write_image(webster_phase_path, width=1200, height=400, scale=2)
            print(f"  Webster phase diagram saved: {webster_phase_path}")
        except Exception as e:
            print(f"  Warning: Could not generate phase diagrams: {e}")
        else:
        print(f"  Warning: Plotly not available, skipping phase diagrams")
    
    return {
        'ml_plan': ml_plan,
        'webster_plan': webster_plan,
        'ml_metrics': ml_metrics,
        'webster_metrics': webster_metrics
    }


def main():
    """Main pipeline orchestrator"""
    print("="*80)
    print("ML vs WEBSTER COMPARISON PIPELINE")
    print("="*80)
    
    # Step 1: Generate training CSV once
    print("\n[STEP 1] Generating training CSV...")
    csv_path = 'outputs/synthetic_training_data.csv'
    final.generate_and_save_training_csv(csv_path)
    
    # Step 2: Train ML models once
    print("\n[STEP 2] Training ML models...")
    cycle_model, green_model, cycle_features, green_features = final.train_models_from_csv(csv_path)
    print("  Models trained successfully")
    
    # Step 3: Process each intersection
    results = {}
    for intersection_num, pcu_dict in [(1, INTERSECTION_1_PCU), (2, INTERSECTION_2_PCU)]:
        results[intersection_num] = run_intersection_pipeline(
            intersection_num, pcu_dict, cycle_model, green_model, cycle_features, green_features
        )
    
    # Step 4: Generate combined comparison charts
    if results[1]['ml_metrics'] and results[1]['webster_metrics'] and \
       results[2]['ml_metrics'] and results[2]['webster_metrics']:
        print("\nGenerating combined comparison charts...")
        combined_output_dir = os.path.join('report', 'images')
        generate_combined_comparison_charts(
            results[1]['ml_metrics'], results[1]['webster_metrics'],
            results[2]['ml_metrics'], results[2]['webster_metrics'],
            combined_output_dir
        )
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - COMPREHENSIVE SUMMARY")
    print("="*80)
    
    intersection_names = {1: "Jyoti Circle (T-junction)", 2: "Hampankatta Circle (4-approach)"}
    
    for intersection_num in [1, 2]:
        res = results[intersection_num]
        int_name = intersection_names[intersection_num]
        print(f"\n{'='*80}")
        print(f"INTERSECTION {intersection_num}: {int_name}")
        print(f"{'='*80}")
        
        if res['ml_metrics'] and res['webster_metrics']:
            ml_metrics = res['ml_metrics']
            webster_metrics = res['webster_metrics']
            ml_plan = res['ml_plan']
            webster_plan = res['webster_plan']
            
            print(f"\nML-BASED METHOD:")
            print(f"  SUMO Metrics:")
            print(f"    Vehicle Count:              {ml_metrics.get('vehicle_count', 0):,} vehicles")
            print(f"    Average Delay:              {ml_metrics.get('avg_delay', 0):.2f} seconds")
            print(f"    Average Waiting Time:       {ml_metrics.get('avg_waiting_time', 0):.2f} seconds")
            print(f"    Average Travel Time:        {ml_metrics.get('avg_travel_time', 0):.2f} seconds")
            print(f"    Average Time Loss:          {ml_metrics.get('avg_time_loss', 0):.2f} seconds")
            print(f"    Average Departure Delay:    {ml_metrics.get('avg_depart_delay', 0):.2f} seconds")
            print(f"    Total Delay:                {ml_metrics.get('total_delay', 0):,.0f} seconds")
            print(f"    Total Waiting Time:         {ml_metrics.get('total_waiting_time', 0):,.0f} seconds")
            print(f"  Signal Plan:")
            print(f"    Cycle Length:               {ml_plan['cycle_length']:.2f} seconds")
            for appr_id, appr_data in ml_plan.get('approaches', {}).items():
                if appr_data.get('green', 0) > 0:
                    print(f"    {appr_id}: Green={appr_data.get('green', 0):.2f}s, "
                          f"Amber={appr_data.get('amber', 0):.2f}s, "
                          f"Red={appr_data.get('red', 0):.2f}s, "
                          f"Flow={appr_data.get('arrival_flow_rate', 0):.0f} PCU/hr")
            
            print(f"\nWEBSTER-BASED METHOD:")
            print(f"  SUMO Metrics:")
            print(f"    Vehicle Count:              {webster_metrics.get('vehicle_count', 0):,} vehicles")
            print(f"    Average Delay:              {webster_metrics.get('avg_delay', 0):.2f} seconds")
            print(f"    Average Waiting Time:       {webster_metrics.get('avg_waiting_time', 0):.2f} seconds")
            print(f"    Average Travel Time:        {webster_metrics.get('avg_travel_time', 0):.2f} seconds")
            print(f"    Average Time Loss:          {webster_metrics.get('avg_time_loss', 0):.2f} seconds")
            print(f"    Average Departure Delay:    {webster_metrics.get('avg_depart_delay', 0):.2f} seconds")
            print(f"    Total Delay:                {webster_metrics.get('total_delay', 0):,.0f} seconds")
            print(f"    Total Waiting Time:         {webster_metrics.get('total_waiting_time', 0):,.0f} seconds")
            print(f"  Signal Plan:")
            print(f"    Cycle Length:               {webster_plan['cycle_length']:.2f} seconds")
            for appr_id, appr_data in webster_plan.get('approaches', {}).items():
                if appr_data.get('green', 0) > 0:
                    print(f"    {appr_id}: Green={appr_data.get('green', 0):.2f}s, "
                          f"Amber={appr_data.get('amber', 0):.2f}s, "
                          f"Red={appr_data.get('red', 0):.2f}s, "
                          f"Flow={appr_data.get('arrival_flow_rate', 0):.0f} PCU/hr")
            
            print(f"\nCOMPARISON:")
            ml_delay = ml_metrics.get('avg_delay', 0)
            webster_delay = webster_metrics.get('avg_delay', 0)
            delay_diff = ml_delay - webster_delay
            delay_pct = (delay_diff / webster_delay * 100) if webster_delay > 0 else 0
            delay_winner = "ML" if ml_delay < webster_delay else "Webster"
            
            print(f"  Average Delay:")
            print(f"    ML:      {ml_delay:.2f} seconds")
            print(f"    Webster: {webster_delay:.2f} seconds")
            print(f"    Difference: {delay_diff:+.2f} seconds ({delay_pct:+.1f}%)")
            print(f"    → Winner: {delay_winner}-based method")
            
            ml_travel = ml_metrics.get('avg_travel_time', 0)
            webster_travel = webster_metrics.get('avg_travel_time', 0)
            travel_diff = ml_travel - webster_travel
            travel_pct = (travel_diff / webster_travel * 100) if webster_travel > 0 else 0
            travel_winner = "ML" if ml_travel < webster_travel else "Webster"
            
            print(f"\n  Average Travel Time:")
            print(f"    ML:      {ml_travel:.2f} seconds")
            print(f"    Webster: {webster_travel:.2f} seconds")
            print(f"    Difference: {travel_diff:+.2f} seconds ({travel_pct:+.1f}%)")
            print(f"    → Winner: {travel_winner}-based method")
            
            ml_time_loss = ml_metrics.get('avg_time_loss', 0)
            webster_time_loss = webster_metrics.get('avg_time_loss', 0)
            time_loss_diff = ml_time_loss - webster_time_loss
            time_loss_pct = (time_loss_diff / webster_time_loss * 100) if webster_time_loss > 0 else 0
            time_loss_winner = "ML" if ml_time_loss < webster_time_loss else "Webster"
            
            print(f"\n  Average Time Loss:")
            print(f"    ML:      {ml_time_loss:.2f} seconds")
            print(f"    Webster: {webster_time_loss:.2f} seconds")
            print(f"    Difference: {time_loss_diff:+.2f} seconds ({time_loss_pct:+.1f}%)")
            print(f"    → Winner: {time_loss_winner}-based method")
            
            ml_throughput = ml_metrics.get('vehicle_count', 0)
            webster_throughput = webster_metrics.get('vehicle_count', 0)
            throughput_diff = ml_throughput - webster_throughput
            throughput_pct = (throughput_diff / webster_throughput * 100) if webster_throughput > 0 else 0
            throughput_winner = "ML" if ml_throughput > webster_throughput else "Webster"
            
            print(f"\n  Throughput:")
            print(f"    ML:      {ml_throughput:,} vehicles")
            print(f"    Webster: {webster_throughput:,} vehicles")
            print(f"    Difference: {throughput_diff:+,} vehicles ({throughput_pct:+.1f}%)")
            print(f"    → Winner: {throughput_winner}-based method (higher is better)")
            
            cycle_diff = ml_plan['cycle_length'] - webster_plan['cycle_length']
            cycle_winner = "ML" if abs(cycle_diff) < 1.0 else ("ML" if ml_plan['cycle_length'] < webster_plan['cycle_length'] else "Webster")
            
            print(f"\n  Cycle Length:")
            print(f"    ML:      {ml_plan['cycle_length']:.2f} seconds")
            print(f"    Webster: {webster_plan['cycle_length']:.2f} seconds")
            print(f"    Difference: {cycle_diff:+.2f} seconds")
            if abs(cycle_diff) < 1.0:
                print(f"    → Similar cycle lengths (difference < 1s)")
            else:
                print(f"    → Shorter cycle: {cycle_winner}-based method")
        
        print(f"\n  Charts saved to: report/images/intersection {intersection_num}/")
        print(f"  Phase diagrams saved to: report/images/intersection {intersection_num}/")
    
    print(f"\n{'='*80}")
    print(f"COMBINED COMPARISON CHARTS:")
    print(f"  report/images/combined_time_metrics_comparison.png")
    print(f"  report/images/combined_traffic_throughput_comparison.png")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
