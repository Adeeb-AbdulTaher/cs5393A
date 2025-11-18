"""
Generate tables for combined multi-bug data
Creates CSV tables for analysis and reporting
"""
import numpy as np
import pandas as pd
import os
import json

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 60)
print("Generating Tables for Combined Data")
print("=" * 60)

# Load data
print("\nLoading data...")
try:
    X = np.load(os.path.join(OUTPUT_DIR, 'combined_coverage_matrix.npy'))
    y = np.load(os.path.join(OUTPUT_DIR, 'combined_test_labels.npy'))
    bug_info = pd.read_csv(os.path.join(OUTPUT_DIR, 'combined_bug_info.csv'))
    metadata = json.load(open(os.path.join(OUTPUT_DIR, 'combined_metadata.json')))
    print(f"  ✓ Loaded data")
except Exception as e:
    print(f"  ✗ Error loading data: {e}")
    print("  Make sure to run combine_multi_bug_data.py first!")
    exit(1)

# 1. Summary Statistics Table
print("\n[1/5] Creating summary statistics table...")
summary_data = {
    'Metric': [
        'Total Bugs',
        'Total Tests',
        'Failing Tests',
        'Passing Tests',
        'Failing Percentage',
        'Code Units (Lines)',
        'Average Tests per Bug',
        'Average Failing per Bug',
        'Average Coverage per Test',
        'Total Coverage Matrix Size'
    ],
    'Value': [
        metadata['num_bugs'],
        metadata['total_tests'],
        metadata['total_failing'],
        metadata['total_passing'],
        f"{metadata['failing_percentage']:.2f}%",
        metadata['code_units'],
        f"{metadata['total_tests'] / metadata['num_bugs']:.1f}",
        f"{metadata['total_failing'] / metadata['num_bugs']:.2f}",
        f"{X.sum(axis=1).mean():.1f}",
        f"{X.shape[0]} × {X.shape[1]}"
    ]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(TABLES_DIR, 'table_summary_statistics.csv'), index=False)
print(f"  ✓ Saved: tables/table_summary_statistics.csv")

# 2. Bug-by-Bug Analysis Table
print("\n[2/5] Creating bug-by-bug analysis table...")
bug_analysis = bug_info.copy()

# Calculate passing tests if not present
if 'passing' not in bug_analysis.columns:
    bug_analysis['passing'] = bug_analysis['tests'] - bug_analysis['failing']

# Fill missing bug names
if 'bug' in bug_analysis.columns:
    bug_analysis['bug'] = bug_analysis['bug'].fillna('')
    # If bug column is empty, create bug names from index
    if bug_analysis['bug'].str.strip().eq('').all():
        bug_analysis['bug'] = [f"Chart_{i+1}" for i in range(len(bug_analysis))]
else:
    bug_analysis['bug'] = [f"Chart_{i+1}" for i in range(len(bug_analysis))]

bug_analysis['failing_rate'] = (bug_analysis['failing'] / bug_analysis['tests'] * 100).round(2)
bug_analysis['passing_rate'] = (bug_analysis['passing'] / bug_analysis['tests'] * 100).round(2)
bug_analysis = bug_analysis[['bug', 'tests', 'failing', 'passing', 'failing_rate', 'passing_rate', 'lines']]
bug_analysis.columns = ['Bug', 'Total Tests', 'Failing', 'Passing', 'Failing %', 'Passing %', 'Code Lines']
bug_analysis.to_csv(os.path.join(TABLES_DIR, 'table_bug_analysis.csv'), index=False)
print(f"  ✓ Saved: tables/table_bug_analysis.csv")

# 3. Coverage Statistics Table
print("\n[3/5] Creating coverage statistics table...")
coverage_per_test = X.sum(axis=1)
failing_coverage = coverage_per_test[y == 1]
passing_coverage = coverage_per_test[y == 0]

coverage_stats = {
    'Category': ['All Tests', 'Failing Tests', 'Passing Tests'],
    'Count': [
        len(coverage_per_test),
        len(failing_coverage),
        len(passing_coverage)
    ],
    'Mean Coverage': [
        f"{coverage_per_test.mean():.2f}",
        f"{failing_coverage.mean():.2f}" if len(failing_coverage) > 0 else "N/A",
        f"{passing_coverage.mean():.2f}"
    ],
    'Median Coverage': [
        f"{np.median(coverage_per_test):.2f}",
        f"{np.median(failing_coverage):.2f}" if len(failing_coverage) > 0 else "N/A",
        f"{np.median(passing_coverage):.2f}"
    ],
    'Min Coverage': [
        f"{coverage_per_test.min()}",
        f"{failing_coverage.min()}" if len(failing_coverage) > 0 else "N/A",
        f"{passing_coverage.min()}"
    ],
    'Max Coverage': [
        f"{coverage_per_test.max()}",
        f"{failing_coverage.max()}" if len(failing_coverage) > 0 else "N/A",
        f"{passing_coverage.max()}"
    ]
}
coverage_df = pd.DataFrame(coverage_stats)
coverage_df.to_csv(os.path.join(TABLES_DIR, 'table_coverage_statistics.csv'), index=False)
print(f"  ✓ Saved: tables/table_coverage_statistics.csv")

# 4. Class Imbalance Analysis Table
print("\n[4/5] Creating class imbalance analysis table...")
imbalance_data = {
    'Metric': [
        'Failing Tests',
        'Passing Tests',
        'Total Tests',
        'Failing Percentage',
        'Passing Percentage',
        'Imbalance Ratio (Passing:Failing)',
        'Class Imbalance Severity'
    ],
    'Value': [
        metadata['total_failing'],
        metadata['total_passing'],
        metadata['total_tests'],
        f"{metadata['failing_percentage']:.2f}%",
        f"{(metadata['total_passing'] / metadata['total_tests'] * 100):.2f}%",
        f"{metadata['total_passing'] / metadata['total_failing']:.1f}:1" if metadata['total_failing'] > 0 else "N/A",
        'Severe' if metadata['failing_percentage'] < 1 else 'Moderate' if metadata['failing_percentage'] < 5 else 'Mild'
    ]
}
imbalance_df = pd.DataFrame(imbalance_data)
imbalance_df.to_csv(os.path.join(TABLES_DIR, 'table_class_imbalance.csv'), index=False)
print(f"  ✓ Saved: tables/table_class_imbalance.csv")

# 5. Comparison Table (Before vs After)
print("\n[5/5] Creating before/after comparison table...")
comparison_data = {
    'Metric': [
        'Number of Bugs',
        'Total Tests',
        'Failing Tests',
        'Passing Tests',
        'Failing Percentage',
        'Code Units',
        'Average Tests per Bug',
        'Average Failing per Bug'
    ],
    'Before (Single Bug)': [
        '1',
        '2,193',
        '1',
        '2,192',
        '0.046%',
        '574',
        '2,193.0',
        '1.0'
    ],
    'After (Combined)': [
        str(metadata['num_bugs']),
        f"{metadata['total_tests']:,}",
        str(metadata['total_failing']),
        f"{metadata['total_passing']:,}",
        f"{metadata['failing_percentage']:.2f}%",
        f"{metadata['code_units']:,}",
        f"{metadata['total_tests'] / metadata['num_bugs']:.1f}",
        f"{metadata['total_failing'] / metadata['num_bugs']:.2f}"
    ],
    'Improvement': [
        f"{metadata['num_bugs']}x",
        f"{metadata['total_tests'] / 2193:.1f}x",
        f"{metadata['total_failing']}x",
        f"{metadata['total_passing'] / 2192:.1f}x",
        f"{metadata['failing_percentage'] / 0.046:.1f}x",
        f"{metadata['code_units'] / 574:.1f}x",
        f"{(metadata['total_tests'] / metadata['num_bugs']) / 2193:.2f}x",
        f"{(metadata['total_failing'] / metadata['num_bugs']) / 1:.1f}x"
    ]
}
comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(os.path.join(TABLES_DIR, 'table_before_after_comparison.csv'), index=False)
print(f"  ✓ Saved: tables/table_before_after_comparison.csv")

print(f"\n{'='*60}")
print("✓ All tables generated successfully!")
print(f"{'='*60}")
print(f"\nTables saved to: {TABLES_DIR}/")
print("  - table_summary_statistics.csv")
print("  - table_bug_analysis.csv")
print("  - table_coverage_statistics.csv")
print("  - table_class_imbalance.csv")
print("  - table_before_after_comparison.csv")

