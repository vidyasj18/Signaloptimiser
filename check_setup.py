"""
Setup Verification Script

Checks if all components are properly installed and configured
for the Signal Optimization Pipeline.
"""

import sys
import os
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def check_python():
    """Check Python version"""
    print("\n[1/6] Checking Python...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_sumo():
    """Check SUMO installation"""
    print("\n[2/6] Checking SUMO...")
    try:
        result = subprocess.run(['sumo', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"  ✅ SUMO installed: {version_line}")
            
            # Check netconvert
            result2 = subprocess.run(['netconvert', '--version'], 
                                   capture_output=True, text=True, timeout=5)
            if result2.returncode == 0:
                print(f"  ✅ netconvert available")
                return True
            else:
                print(f"  ⚠️ netconvert not found")
                return False
        else:
            print(f"  ❌ SUMO not working properly")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"  ❌ SUMO not found in PATH")
        print(f"  Download from: https://sumo.dlr.de/docs/Downloads.php")
        return False

def check_output_dir():
    """Check output directory structure"""
    print("\n[3/6] Checking output directories...")
    dirs = ['outputs', 'outputs/visualizations', 'report', 'report/images']
    all_ok = True
    
    for dir_path in dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ⚠️ {dir_path}/ (will be created)")
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                print(f"     Created {dir_path}/")
            except Exception as e:
                print(f"     ❌ Failed to create: {e}")
                all_ok = False
    
    return all_ok

def check_data_files():
    """Check if required data files exist"""
    print("\n[4/6] Checking data files...")
    files = {
        'outputs/intersection_summary.json': 'PCU data',
        'outputs/webster_signal_plan.json': 'Webster signal plan',
        'outputs/sumo_metrics_webster.json': 'SUMO metrics (Webster)',
        'yolov8n.pt': 'YOLO model'
    }
    
    all_exist = True
    for filepath, description in files.items():
        if Path(filepath).exists():
            size = Path(filepath).stat().st_size / 1024
            print(f"  ✅ {description}: {filepath} ({size:.1f} KB)")
        else:
            print(f"  ⚠️ {description}: {filepath} (not found)")
            if filepath == 'yolov8n.pt':
                print(f"     Will download automatically on first run")
            else:
                print(f"     Run required scripts to generate")
                all_exist = False
    
    return all_exist

def check_scripts():
    """Check if all main scripts exist"""
    print("\n[5/6] Checking main scripts...")
    scripts = [
        'webster.py',
        'sumo_simulation.py',
        'generate_visualizations.py'
    ]
    
    all_ok = True
    for script in scripts:
        if Path(script).exists():
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script} (MISSING)")
            all_ok = False
    
    return all_ok

def check_sumo_files():
    """Check SUMO generated files"""
    print("\n[6/6] Checking SUMO files...")
    sumo_files = [
        'intersection.nod.xml',
        'intersection.edg.xml',
        'sumo_network.net.xml',
        'webster_traffic_lights.add.xml',
        'routes.rou.xml',
        'webster_config.sumocfg'
    ]
    
    exist_count = 0
    for filepath in sumo_files:
        if Path(filepath).exists():
            print(f"  ✅ {filepath}")
            exist_count += 1
        else:
            print(f"  ⚠️ {filepath} (not generated yet)")
    
    if exist_count == 0:
        print(f"\n  Run 'python sumo_simulation.py' to generate SUMO files")
    elif exist_count < len(sumo_files):
        print(f"\n  Some SUMO files missing - may need regeneration")
    
    return True  # Not critical, can be generated

def main():
    print_header("SIGNAL OPTIMIZATION PIPELINE - SETUP VERIFICATION")
    
    results = {
        'Python': check_python(),
        'SUMO': check_sumo(),
        'Directories': check_output_dir(),
        'Data Files': check_data_files(),
        'Scripts': check_scripts(),
        'SUMO Files': check_sumo_files()
    }
    
    print_header("SETUP VERIFICATION SUMMARY")
    
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component}")
    
    critical = ['Python', 'Scripts']
    critical_ok = all(results[k] for k in critical)
    
    print("\n" + "=" * 80)
    if critical_ok:
        print("  ✅ CRITICAL COMPONENTS: ALL OK")
        print("=" * 80)
        print("\n🎉 Your setup is ready!")
        print("\nNext steps:")
        if not results['Data Files']:
            print("  1. Ensure outputs/intersection_summary.json exists.")
            print("  2. Run: python webster.py")
        if results['SUMO'] and not results['SUMO Files']:
            print("  3. Run: python sumo_simulation.py")
        print("  4. Run: python generate_visualizations.py (runs both steps)")
    else:
        print("  ❌ CRITICAL ISSUES DETECTED")
        print("=" * 80)
        print("\n⚠️ Please fix the issues above before proceeding.")
        missing_critical = [k for k in critical if not results[k]]
        if missing_critical:
            print(f"\nMissing critical components: {', '.join(missing_critical)}")
    
    print("\n" + "=" * 80)
    
    return 0 if critical_ok else 1

if __name__ == "__main__":
    sys.exit(main())

