"""Verify combined data and retrain models"""
import numpy as np
import os

print("=" * 70)
print("Verifying Combined Data")
print("=" * 70)

# Check combined data
if os.path.exists('combined/combined_coverage_matrix.npy'):
    X = np.load('combined/combined_coverage_matrix.npy')
    y = np.load('combined/combined_test_labels.npy')
    
    print(f"\nCombined Data:")
    print(f"  Matrix shape: {X.shape}")
    print(f"  Total tests: {len(y):,}")
    print(f"  Failing tests: {y.sum():,} ({y.sum()/len(y)*100:.3f}%)")
    print(f"  Passing tests: {len(y)-y.sum():,}")
    
    # Check individual bugs
    import glob
    bug_dirs = glob.glob("multi_bug_data/Chart_*")
    print(f"\nIndividual Bugs Found: {len(bug_dirs)}")
    
    total_individual_failing = 0
    for bug_dir in sorted(bug_dirs):
        failing_file = os.path.join(bug_dir, "failing_tests")
        if os.path.exists(failing_file):
            with open(failing_file, 'r') as f:
                content = f.read()
                count = content.count('---')
                total_individual_failing += count
    
    print(f"  Total failing tests in individual bugs: {total_individual_failing}")
    print(f"  Combined dataset failing tests: {y.sum()}")
    
    if y.sum() < total_individual_failing:
        print(f"\n  ⚠️  Warning: Combined has fewer failing tests than individual bugs")
        print(f"     This might be due to test deduplication or combination logic")
    else:
        print(f"\n  ✓ Data looks consistent")
else:
    print("  ✗ Combined data not found! Run combine_multi_bug_data.py first")

print("\n" + "=" * 70)

