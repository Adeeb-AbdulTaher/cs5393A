"""
Run all scripts in sequence with explicit output
"""
import subprocess
import sys
import os

scripts = [
    'combine_multi_bug_data.py',
    'generate_graphs.py', 
    'generate_tables.py',
    'generate_report.py'
]

print("=" * 60)
print("Running All Combined Data Scripts")
print("=" * 60)

for script in scripts:
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            print(f"ERROR: Script {script} failed with return code {result.returncode}")
        else:
            print(f"✓ {script} completed successfully")
    except Exception as e:
        print(f"✗ Error running {script}: {e}")

print(f"\n{'='*60}")
print("All scripts completed!")
print(f"{'='*60}")

