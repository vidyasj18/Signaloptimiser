"""
Signal Plan Comparison Script

This script reads signal timing data from two JSON files (ML-based and Webster-based)
and calculates performance metrics (delay, queue length, throughput, LOS) for each approach,
then compares the two plans to determine which performs better.
"""

import json
import os
import math
from typing import Dict, List, Tuple


def calculate_delay(c: float, g: float, q: float, s: float) -> float:
    """
    Calculate average vehicle delay using Webster's delay formula.
    
    Parameters:
    - c: Cycle length (seconds)
    - g: Effective green time (seconds)
    - q: Arrival flow rate (vehicles/hour)
    - s: Saturation flow rate (vehicles/hour)
    
    Returns:
    - Average delay per vehicle (seconds)
    """
    if s <= 0 or c <= 0:
        return float('inf')
    
    X = q / s  # volume-to-capacity ratio
    
    if X >= 1:
        return float('inf')  # oversaturated condition, infinite delay
    
    if g <= 0:
        return float('inf')  # no green time
    
    # Webster's delay formula
    # Uniform delay component
    uniform_delay = (c * (1 - g / c) ** 2) / (2 * (1 - X * (g / c)))
    
    # Random delay component (simplified)
    if X > 0 and (1 - X) > 0:
        random_delay = (X ** 2) / (2 * q * (1 - X))
    else:
        random_delay = 0.0
    
    total_delay = uniform_delay + random_delay
    
    # Handle edge cases
    if not math.isfinite(total_delay) or total_delay < 0:
        return float('inf')
    
    return total_delay


def calculate_queue_length(delay: float, q: float) -> float:
    """
    Calculate average queue length in vehicles.
    
    Parameters:
    - delay: Average delay per vehicle (seconds)
    - q: Arrival flow rate (vehicles/hour)
    
    Returns:
    - Average queue length (vehicles)
    """
    if delay == float('inf') or q <= 0:
        return float('inf')
    # Queue length = (arrival rate * delay) / 3600
    return (q * delay) / 3600.0


def calculate_throughput(g: float, c: float, s: float, T: float = 1.0) -> float:
    """
    Calculate throughput (vehicles discharged per hour).
    
    Parameters:
    - g: Effective green time (seconds)
    - c: Cycle length (seconds)
    - s: Saturation flow rate (vehicles/hour)
    - T: Time period in hours (default 1 hour)
    
    Returns:
    - Throughput (vehicles/hour)
    """
    if c <= 0 or s <= 0:
        return 0.0
    # Throughput = saturation flow * (green time / cycle length)
    return s * (g / c) * T


def calculate_los(delay: float) -> str:
    """
    Calculate Level of Service (LOS) based on average delay.
    
    LOS thresholds (seconds):
    - A: <= 10
    - B: <= 20
    - C: <= 35
    - D: <= 55
    - E: <= 80
    - F: > 80 or infinite
    """
    if delay == float('inf') or not math.isfinite(delay):
        return 'F'
    elif delay <= 10:
        return 'A'
    elif delay <= 20:
        return 'B'
    elif delay <= 35:
        return 'C'
    elif delay <= 55:
        return 'D'
    elif delay <= 80:
        return 'E'
    else:
        return 'F'


def evaluate_signal_plan(plan_data: Dict) -> Dict[str, Dict]:
    """
    Evaluate a signal plan and calculate metrics for each approach.
    
    Parameters:
    - plan_data: Dictionary containing cycle_length and approaches data
    
    Returns:
    - Dictionary with metrics for each approach
    """
    c = plan_data['cycle_length']
    results = {}
    
    for approach_name, approach_data in plan_data['approaches'].items():
        g = approach_data['effective_green']
        q = approach_data['arrival_flow_rate']
        s = approach_data['saturation_flow_rate']
        
        delay = calculate_delay(c, g, q, s)
        queue = calculate_queue_length(delay, q)
        throughput = calculate_throughput(g, c, s)
        los = calculate_los(delay)
        
        results[approach_name] = {
            'delay': delay,
            'queue_length': queue,
            'throughput': throughput,
            'LOS': los,
            'cycle_length': c,
            'effective_green': g,
            'arrival_flow_rate': q,
            'saturation_flow_rate': s
        }
    
    return results


def compare_plans(plan1_results: Dict[str, Dict], plan2_results: Dict[str, Dict],
                  plan1_name: str = "Plan 1", plan2_name: str = "Plan 2") -> None:
    """
    Compare two signal plans and print detailed comparison.
    
    Parameters:
    - plan1_results: Results dictionary from first plan
    - plan2_results: Results dictionary from second plan
    - plan1_name: Name of first plan
    - plan2_name: Name of second plan
    """
    print("=" * 80)
    print(f"COMPARING SIGNAL PLANS: {plan1_name} vs {plan2_name}")
    print("=" * 80)
    
    # Get all approaches present in either plan
    all_approaches = set(plan1_results.keys()) | set(plan2_results.keys())
    
    if not all_approaches:
        print("ERROR: No approaches found in either plan!")
        return
    
    metrics = ['delay', 'queue_length', 'throughput']
    better = {plan1_name: 0, plan2_name: 0}
    approach_comparisons = {}
    
    print(f"\n{'Approach':<8} {'Metric':<20} {plan1_name:<25} {plan2_name:<25} {'Better':<10}")
    print("-" * 80)
    
    for approach in sorted(all_approaches):
        approach_comparisons[approach] = {}
        
        if approach not in plan1_results:
            print(f"{approach:<8} {'N/A in Plan 1':<20} {'N/A':<25} {plan2_name:<25} {plan2_name}")
            better[plan2_name] += len(metrics)
            continue
        
        if approach not in plan2_results:
            print(f"{approach:<8} {'N/A in Plan 2':<20} {plan1_name:<25} {'N/A':<25} {plan1_name}")
            better[plan1_name] += len(metrics)
            continue
        
        p1 = plan1_results[approach]
        p2 = plan2_results[approach]
        
        for metric in metrics:
            v1 = p1[metric]
            v2 = p2[metric]
            
            # Determine which is better
            if metric == 'throughput':
                # Higher is better
                if v1 == float('inf') and v2 == float('inf'):
                    better_plan = "Tie"
                elif v1 == float('inf'):
                    better_plan = plan1_name
                    better[plan1_name] += 1
                elif v2 == float('inf'):
                    better_plan = plan2_name
                    better[plan2_name] += 1
                elif v1 > v2:
                    better_plan = plan1_name
                    better[plan1_name] += 1
                elif v2 > v1:
                    better_plan = plan2_name
                    better[plan2_name] += 1
                else:
                    better_plan = "Tie"
            else:
                # Lower is better (delay, queue_length)
                if v1 == float('inf') and v2 == float('inf'):
                    better_plan = "Tie"
                elif v1 == float('inf'):
                    better_plan = plan2_name
                    better[plan2_name] += 1
                elif v2 == float('inf'):
                    better_plan = plan1_name
                    better[plan1_name] += 1
                elif v1 < v2:
                    better_plan = plan1_name
                    better[plan1_name] += 1
                elif v2 < v1:
                    better_plan = plan2_name
                    better[plan2_name] += 1
                else:
                    better_plan = "Tie"
            
            # Format values for display
            if v1 == float('inf'):
                v1_str = "Inf"
            else:
                v1_str = f"{v1:.2f}"
            
            if v2 == float('inf'):
                v2_str = "Inf"
            else:
                v2_str = f"{v2:.2f}"
            
            print(f"{approach:<8} {metric.capitalize():<20} {v1_str:<25} {v2_str:<25} {better_plan:<10}")
            
            approach_comparisons[approach][metric] = {
                plan1_name: v1,
                plan2_name: v2,
                'better': better_plan
            }
        
        # LOS comparison
        los1 = p1['LOS']
        los2 = p2['LOS']
        los_order = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
        if los_order[los1] < los_order[los2]:
            better_los = plan1_name
            better[plan1_name] += 1
        elif los_order[los2] < los_order[los1]:
            better_los = plan2_name
            better[plan2_name] += 1
        else:
            better_los = "Tie"
        
        print(f"{approach:<8} {'LOS':<20} {los1:<25} {los2:<25} {better_los:<10}")
        print("-" * 80)
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"{plan1_name} wins: {better[plan1_name]} metrics")
    print(f"{plan2_name} wins: {better[plan2_name]} metrics")
    
    if better[plan1_name] > better[plan2_name]:
        print(f"\n>>> OVERALL BETTER PLAN: {plan1_name} <<<")
    elif better[plan2_name] > better[plan1_name]:
        print(f"\n>>> OVERALL BETTER PLAN: {plan2_name} <<<")
    else:
        print(f"\n>>> OVERALL: TIE <<<")
    
    # Average metrics comparison
    print("\n" + "=" * 80)
    print("AVERAGE METRICS (across all approaches)")
    print("=" * 80)
    
    for metric in metrics:
        p1_values = [p[metric] for p in plan1_results.values() if p[metric] != float('inf')]
        p2_values = [p[metric] for p in plan2_results.values() if p[metric] != float('inf')]
        
        if p1_values:
            avg1 = sum(p1_values) / len(p1_values)
        else:
            avg1 = float('inf')
        
        if p2_values:
            avg2 = sum(p2_values) / len(p2_values)
        else:
            avg2 = float('inf')
        
        if metric == 'throughput':
            diff = avg1 - avg2 if avg1 != float('inf') and avg2 != float('inf') else 0
            diff_pct = (diff / avg2 * 100) if avg2 != float('inf') and avg2 > 0 else 0
        else:
            diff = avg2 - avg1 if avg1 != float('inf') and avg2 != float('inf') else 0
            diff_pct = (diff / avg1 * 100) if avg1 != float('inf') and avg1 > 0 else 0
        
        avg1_str = f"{avg1:.2f}" if avg1 != float('inf') else "Inf"
        avg2_str = f"{avg2:.2f}" if avg2 != float('inf') else "Inf"
        diff_str = f"{diff:.2f}" if diff != float('inf') and diff != float('-inf') else "N/A"
        diff_pct_str = f"{diff_pct:.1f}%" if diff_pct != float('inf') and diff_pct != float('-inf') else "N/A"
        
        print(f"{metric.capitalize():<20} {plan1_name}: {avg1_str:<15} {plan2_name}: {avg2_str:<15} "
              f"Difference: {diff_str} ({diff_pct_str})")


def load_signal_plan(filepath: str) -> Dict:
    """
    Load signal plan data from JSON file.
    
    Parameters:
    - filepath: Path to JSON file
    
    Returns:
    - Dictionary containing signal plan data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Signal plan file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """
    Main function to run the comparison.
    """
    # Default file paths
    ml_plan_path = 'outputs/ml_signal_plan.json'
    webster_plan_path = 'outputs/webster_signal_plan.json'
    
    # Check if files exist
    if not os.path.exists(ml_plan_path):
        print(f"ERROR: ML signal plan file not found: {ml_plan_path}")
        print("Please run final.py first to generate the ML signal plan.")
        return
    
    if not os.path.exists(webster_plan_path):
        print(f"ERROR: Webster signal plan file not found: {webster_plan_path}")
        print("Please run webster.py first to generate the Webster signal plan.")
        return
    
    # Load both plans
    print("Loading signal plans...")
    ml_plan = load_signal_plan(ml_plan_path)
    webster_plan = load_signal_plan(webster_plan_path)
    
    print(f"Loaded {ml_plan_path}")
    print(f"Loaded {webster_plan_path}\n")
    
    # Evaluate both plans
    print("Evaluating ML-based plan...")
    ml_results = evaluate_signal_plan(ml_plan)
    
    print("Evaluating Webster-based plan...")
    webster_results = evaluate_signal_plan(webster_plan)
    
    # Print individual plan summaries
    print("\n" + "=" * 80)
    print("ML-BASED PLAN SUMMARY")
    print("=" * 80)
    for approach, metrics in ml_results.items():
        print(f"\n{approach}:")
        print(f"  Cycle Length: {metrics['cycle_length']:.2f} s")
        print(f"  Effective Green: {metrics['effective_green']:.2f} s")
        print(f"  Arrival Flow Rate: {metrics['arrival_flow_rate']:.2f} PCU/hr")
        print(f"  Saturation Flow Rate: {metrics['saturation_flow_rate']:.2f} PCU/hr")
        print(f"  Average Delay: {metrics['delay']:.2f} s" if metrics['delay'] != float('inf') else "  Average Delay: Inf s")
        print(f"  Queue Length: {metrics['queue_length']:.2f} vehicles" if metrics['queue_length'] != float('inf') else "  Queue Length: Inf vehicles")
        print(f"  Throughput: {metrics['throughput']:.2f} PCU/hr")
        print(f"  Level of Service: {metrics['LOS']}")
    
    print("\n" + "=" * 80)
    print("WEBSTER-BASED PLAN SUMMARY")
    print("=" * 80)
    for approach, metrics in webster_results.items():
        print(f"\n{approach}:")
        print(f"  Cycle Length: {metrics['cycle_length']:.2f} s")
        print(f"  Effective Green: {metrics['effective_green']:.2f} s")
        print(f"  Arrival Flow Rate: {metrics['arrival_flow_rate']:.2f} PCU/hr")
        print(f"  Saturation Flow Rate: {metrics['saturation_flow_rate']:.2f} PCU/hr")
        print(f"  Average Delay: {metrics['delay']:.2f} s" if metrics['delay'] != float('inf') else "  Average Delay: Inf s")
        print(f"  Queue Length: {metrics['queue_length']:.2f} vehicles" if metrics['queue_length'] != float('inf') else "  Queue Length: Inf vehicles")
        print(f"  Throughput: {metrics['throughput']:.2f} PCU/hr")
        print(f"  Level of Service: {metrics['LOS']}")
    
    # Compare plans
    compare_plans(ml_results, webster_results, "ML-based", "Webster-based")
    
    # Save comparison results to JSON
    comparison_output = {
        "ml_plan": ml_results,
        "webster_plan": webster_results,
        "summary": {
            "ml_plan_file": ml_plan_path,
            "webster_plan_file": webster_plan_path
        }
    }
    
    output_path = 'outputs/signal_plan_comparison.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_output, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Comparison results saved to {output_path} ===")


if __name__ == "__main__":
    main()

