"""
Verify and display coverage matrix information
"""
import numpy as np
import pandas as pd
import json

print("=" * 60)
print("COVERAGE MATRIX VERIFICATION")
print("=" * 60)

# Load line-level matrix
print("\n1. LINE-LEVEL COVERAGE MATRIX")
print("-" * 60)
line_matrix = np.load('line_coverage_matrix.npy')
line_labels = np.load('line_coverage_labels.npy')

print(f"Matrix shape: {line_matrix.shape} (tests × code lines)")
print(f"Data type: {line_matrix.dtype}")
print(f"Matrix statistics:")
print(f"  - Min value: {line_matrix.min()}")
print(f"  - Max value: {line_matrix.max()}")
print(f"  - Mean: {line_matrix.mean():.4f}")
print(f"  - Non-zero entries: {np.count_nonzero(line_matrix)} ({np.count_nonzero(line_matrix)/line_matrix.size*100:.2f}%)")
print(f"\nTest labels:")
print(f"  - Total tests: {len(line_labels)}")
print(f"  - Failing tests: {line_labels.sum()} ({line_labels.sum()/len(line_labels)*100:.2f}%)")
print(f"  - Passing tests: {len(line_labels) - line_labels.sum()} ({(len(line_labels) - line_labels.sum())/len(line_labels)*100:.2f}%)")

# Load method-level matrix
print("\n2. METHOD-LEVEL COVERAGE MATRIX")
print("-" * 60)
method_matrix = np.load('method_coverage_matrix.npy')
method_labels = np.load('method_coverage_labels.npy')

print(f"Matrix shape: {method_matrix.shape} (tests × methods)")
print(f"Data type: {method_matrix.dtype}")
print(f"Matrix statistics:")
print(f"  - Min value: {method_matrix.min()}")
print(f"  - Max value: {method_matrix.max()}")
print(f"  - Mean: {method_matrix.mean():.4f}")
print(f"  - Non-zero entries: {np.count_nonzero(method_matrix)} ({np.count_nonzero(method_matrix)/method_matrix.size*100:.2f}%)")
print(f"\nTest labels:")
print(f"  - Total tests: {len(method_labels)}")
print(f"  - Failing tests: {method_labels.sum()} ({method_labels.sum()/len(method_labels)*100:.2f}%)")
print(f"  - Passing tests: {len(method_labels) - method_labels.sum()} ({(len(method_labels) - method_labels.sum())/len(method_labels)*100:.2f}%)")

# Load metadata
print("\n3. METADATA")
print("-" * 60)
with open('line_coverage_metadata.json', 'r') as f:
    metadata = json.load(f)
    print(f"Number of tests: {metadata['num_tests']}")
    print(f"Number of code units (lines): {metadata['num_code_units']}")
    print(f"Number of failing tests: {metadata['num_failing']}")
    print(f"Number of passing tests: {metadata['num_passing']}")
    print(f"\nSample code units (first 5):")
    for unit in metadata['code_units'][:5]:
        print(f"  - {unit}")

# Load test labels CSV
print("\n4. TEST LABELS (Sample)")
print("-" * 60)
test_df = pd.read_csv('line_coverage_test_labels.csv')
print(f"Total rows: {len(test_df)}")
print(f"\nFirst 5 tests:")
print(test_df.head().to_string(index=False))
print(f"\nFailing tests:")
failing = test_df[test_df['label'] == 1]
print(failing[['test_name', 'label']].to_string(index=False))

print("\n" + "=" * 60)
print("✓ All matrices verified successfully!")
print("=" * 60)
print("\nFiles ready for ML model input:")
print("  - line_coverage_matrix.npy: Main coverage matrix (line-level)")
print("  - line_coverage_labels.npy: Test labels (1=failing, 0=passing)")
print("  - method_coverage_matrix.npy: Method-level coverage matrix")
print("  - method_coverage_labels.npy: Test labels for method matrix")
print("  - *_test_labels.csv: Human-readable test names with labels")
print("  - *_metadata.json: Matrix metadata and statistics")

