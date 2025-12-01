"""
Collect Math bugs (1-20) from Defects4J Math project
"""
import os
import sys
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collect_multiple_bugs import collect_bug_data

print("=" * 70)
print("Collecting Math Bugs (1-20)")
print("=" * 70)
print("Note: Math project has 106 bugs available")
print("This script collects the first 20 for initial testing")
print("=" * 70)

# Collect Math bugs 1-20
bug_ids = list(range(1, 21))
successful = []
failed = []

for bug_id in bug_ids:
    print(f"\n{'='*70}")
    print(f"Collecting Math-{bug_id}...")
    print(f"{'='*70}")
    
    try:
        work_dir = f'/tmp/Math_{bug_id}'
        collect_bug_data('Math', bug_id, work_dir)
        successful.append(bug_id)
        print(f"✓ Math-{bug_id} collected successfully")
    except Exception as e:
        failed.append((bug_id, str(e)))
        print(f"✗ Math-{bug_id} failed: {error}")

print(f"\n{'='*70}")
print("Collection Summary")
print(f"{'='*70}")
print(f"Successful: {len(successful)} bugs - {successful}")
print(f"Failed: {len(failed)} bugs")
if failed:
    for bug_id, error in failed:
        print(f"  - Math-{bug_id}: {error}")

print(f"\n✓ Collection complete!")
print(f"Next step: Run combine_multi_bug_data.py to combine all bugs")
print(f"Or create a new combine script for Math project")

