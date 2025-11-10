"""
Minimal pipeline runner for Webster + SUMO.

Usage:
    python generate_visualizations.py

This script simply executes:
    1) webster.py - to compute the signal plan from intersection data
    2) sumo_simulation.py - to run SUMO and report performance metrics
"""

import subprocess
import sys

STEPS = [
    ("Webster signal optimisation", [sys.executable, "webster.py"]),
    ("SUMO simulation (Webster plan)", [sys.executable, "sumo_simulation.py"]),
]


def run_step(label, command):
    print("\n" + "=" * 80)
    print(label.upper())
    print("=" * 80)

    try:
        result = subprocess.run(command, text=True)
        if result.returncode == 0:
            print(f"\n✅ {label} completed.")
        else:
            print(f"\n⚠️ {label} exited with code {result.returncode}.")
    except Exception as exc:
        print(f"\n❌ Failed to run {command}: {exc}")


def main():
    print("=" * 80)
    print("WEBSTER ➜ SUMO PIPELINE")
    print("=" * 80)

    for label, command in STEPS:
        run_step(label, command)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

