"""
Combine coverage data from multiple bugs into a single dataset
Addresses class imbalance by including failing tests from multiple bugs
All outputs saved to the combined/ folder
"""
import numpy as np
import pandas as pd
import os
import glob
import sys
import re

# Add parent directory to path to import functions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from create_coverage_matrix import parse_coverage_xml, read_test_names, extract_failing_test_names

# Set working directory to parent (where multi_bug_data is)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = SCRIPT_DIR  # Output to combined/ folder

os.chdir(PARENT_DIR)

print("=" * 60)
print("Combining Multi-Bug Coverage Data")
print("=" * 60)

# Find all bug data directories
data_dirs = glob.glob("multi_bug_data/*/")
data_dirs = [d for d in data_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, "coverage.xml"))]

print(f"\nFound {len(data_dirs)} bug directories:")
for d in data_dirs:
    print(f"  - {d}")

if len(data_dirs) == 0:
    print("\nNo bug data found! Make sure multi_bug_data/ folder exists with bug subdirectories.")
    sys.exit(1)

# Collect all coverage matrices and labels
all_matrices = []
all_labels = []
all_test_names = []
bug_info = []

for bug_dir in data_dirs:
    bug_name = os.path.basename(bug_dir.rstrip('/'))
    print(f"\nProcessing {bug_name}...")
    
    coverage_file = os.path.join(bug_dir, "coverage.xml")
    all_tests_file = os.path.join(bug_dir, "all_tests")
    failing_tests_file = os.path.join(bug_dir, "failing_tests")
    
    if not all(os.path.exists(f) for f in [coverage_file, all_tests_file, failing_tests_file]):
        print(f"  ⚠️  Missing files, skipping")
        continue
    
    try:
        # Parse coverage
        line_coverage, classes_info = parse_coverage_xml(coverage_file)
        test_names = read_test_names(all_tests_file)
        failing_tests = extract_failing_test_names(failing_tests_file)
        
        # Create matrix (simplified - using aggregate coverage)
        code_units = sorted(line_coverage.keys())
        num_tests = len(test_names)
        num_units = len(code_units)
        
        # Create matrix
        matrix = np.zeros((num_tests, num_units), dtype=int)
        for j, unit in enumerate(code_units):
            if line_coverage[unit]['hits'] > 0:
                matrix[:, j] = 1
        
        # Create labels
        test_labels = []
        for test in test_names:
            match = re.match(r'(\w+)\(([\w\.]+)\)', test)
            if match:
                test_method = match.group(1)
                test_class = match.group(2)
                test_key = f"{test_class}::{test_method}"
                label = 1 if test_key in failing_tests else 0
            else:
                label = 0
            test_labels.append(label)
        
        all_matrices.append(matrix)
        all_labels.extend(test_labels)
        all_test_names.extend([f"{bug_name}_{test}" for test in test_names])
        
        failing_count = sum(test_labels)
        bug_info.append({
            'bug': bug_name,
            'tests': num_tests,
            'failing': failing_count,
            'passing': num_tests - failing_count,
            'lines': num_units,
            'coverage_rate': f"{num_units/num_tests*100:.1f}%" if num_tests > 0 else "0%"
        })
        
        print(f"  ✓ {num_tests} tests ({failing_count} failing, {num_tests - failing_count} passing), {num_units} code lines")
    except Exception as e:
        print(f"  ✗ Error processing {bug_name}: {str(e)}")
        continue

if len(all_matrices) == 0:
    print("\nNo valid bug data processed!")
    sys.exit(1)

# Combine matrices
print(f"\n{'='*60}")
print("Combining Data")
print(f"{'='*60}")

# Align matrices to same number of code units (use max)
max_units = max(m.shape[1] for m in all_matrices)
print(f"Max code units: {max_units}")

# Pad or truncate matrices to same size
combined_matrices = []
for matrix in all_matrices:
    if matrix.shape[1] < max_units:
        # Pad with zeros
        padding = np.zeros((matrix.shape[0], max_units - matrix.shape[1]), dtype=int)
        matrix = np.hstack([matrix, padding])
    elif matrix.shape[1] > max_units:
        # Truncate
        matrix = matrix[:, :max_units]
    combined_matrices.append(matrix)

# Stack all matrices
X_combined = np.vstack(combined_matrices)
y_combined = np.array(all_labels)

print(f"\nCombined dataset:")
print(f"  Total tests: {len(y_combined):,}")
print(f"  Failing tests: {y_combined.sum()} ({y_combined.sum()/len(y_combined)*100:.2f}%)")
print(f"  Passing tests: {len(y_combined) - y_combined.sum()} ({(len(y_combined) - y_combined.sum())/len(y_combined)*100:.2f}%)")
print(f"  Code units: {X_combined.shape[1]:,}")

# Save combined data to combined/ folder
print(f"\nSaving combined data to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.save(os.path.join(OUTPUT_DIR, 'combined_coverage_matrix.npy'), X_combined)
np.save(os.path.join(OUTPUT_DIR, 'combined_test_labels.npy'), y_combined)

# Save test names
test_df = pd.DataFrame({
    'test_name': all_test_names,
    'label': y_combined,
    'is_failing': y_combined == 1
})
test_df.to_csv(os.path.join(OUTPUT_DIR, 'combined_test_labels.csv'), index=False)

# Save bug info
bug_df = pd.DataFrame(bug_info)
bug_df.to_csv(os.path.join(OUTPUT_DIR, 'combined_bug_info.csv'), index=False)

# Save metadata
metadata = {
    'num_bugs': len(bug_info),
    'total_tests': len(y_combined),
    'total_failing': int(y_combined.sum()),
    'total_passing': int(len(y_combined) - y_combined.sum()),
    'failing_percentage': float(y_combined.sum()/len(y_combined)*100),
    'code_units': int(X_combined.shape[1]),
    'matrix_shape': list(X_combined.shape),
    'bugs': bug_info
}

import json
with open(os.path.join(OUTPUT_DIR, 'combined_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print("  ✓ Saved: combined_coverage_matrix.npy")
print("  ✓ Saved: combined_test_labels.npy")
print("  ✓ Saved: combined_test_labels.csv")
print("  ✓ Saved: combined_bug_info.csv")
print("  ✓ Saved: combined_metadata.json")

print(f"\n{'='*60}")
print("✓ Data combination complete!")
print(f"{'='*60}")
print(f"\nClass balance:")
print(f"  Before (single bug): 1 failing / 2,193 total (0.046%)")
print(f"  After (combined):    {y_combined.sum()} failing / {len(y_combined)} total ({y_combined.sum()/len(y_combined)*100:.2f}%)")
print(f"  Improvement: {y_combined.sum()}x more failing tests!")

print(f"\nNext steps:")
print(f"  1. Run: python combined/generate_graphs.py")
print(f"  2. Run: python combined/generate_tables.py")
print(f"  3. Run: python combined/generate_report.py")

