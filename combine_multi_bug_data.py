"""
Combine coverage data from multiple bugs into a single dataset
Addresses class imbalance by including failing tests from multiple bugs
"""
import numpy as np
import pandas as pd
import os
import glob
import sys
from create_coverage_matrix import parse_coverage_xml, read_test_names, extract_failing_test_names

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
    print("\nNo bug data found! Run collect_multiple_bugs.py first.")
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
    import re
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
        'lines': num_units
    })
    
    print(f"  ✓ {num_tests} tests ({failing_count} failing), {num_units} code lines")

# Combine matrices
print(f"\n{'='*60}")
print("Combining Data")
print(f"{'='*60}")

# For now, we'll pad/align matrices to same number of code units
# In a full implementation, you'd align by actual code line numbers
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
print(f"  Total tests: {len(y_combined)}")
print(f"  Failing tests: {y_combined.sum()} ({y_combined.sum()/len(y_combined)*100:.2f}%)")
print(f"  Passing tests: {len(y_combined) - y_combined.sum()} ({(len(y_combined) - y_combined.sum())/len(y_combined)*100:.2f}%)")
print(f"  Code units: {X_combined.shape[1]}")

# Save combined data
print(f"\nSaving combined data...")
np.save('combined_coverage_matrix.npy', X_combined)
np.save('combined_test_labels.npy', y_combined)

# Save test names
test_df = pd.DataFrame({
    'test_name': all_test_names,
    'label': y_combined
})
test_df.to_csv('combined_test_labels.csv', index=False)

# Save bug info
bug_df = pd.DataFrame(bug_info)
bug_df.to_csv('combined_bug_info.csv', index=False)

print("  ✓ Saved: combined_coverage_matrix.npy")
print("  ✓ Saved: combined_test_labels.npy")
print("  ✓ Saved: combined_test_labels.csv")
print("  ✓ Saved: combined_bug_info.csv")

print(f"\n{'='*60}")
print("✓ Data combination complete!")
print(f"{'='*60}")
print(f"\nClass balance improved:")
print(f"  Before: 1 failing / 2,193 total (0.046%)")
print(f"  After:  {y_combined.sum()} failing / {len(y_combined)} total ({y_combined.sum()/len(y_combined)*100:.2f}%)")
print(f"\nYou can now retrain the model with:")
print(f"  X = np.load('combined_coverage_matrix.npy')")
print(f"  y = np.load('combined_test_labels.npy')")

