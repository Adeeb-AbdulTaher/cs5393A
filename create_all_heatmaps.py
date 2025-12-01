"""
Create individual heatmaps for each bug and combined heatmap
Simplified version with explicit output
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from create_coverage_matrix import parse_coverage_xml, read_test_names, extract_failing_test_names

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

OUTPUT_DIR = os.path.join('combined', 'visuals')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("Generating Individual Bug Heatmaps")
print("=" * 70)

# Process each bug individually
bugs = ['Chart_1', 'Chart_2', 'Chart_3', 'Chart_4', 'Chart_5']
created_heatmaps = []

for bug_name in bugs:
    print(f"\n{'='*70}")
    print(f"Processing {bug_name}...")
    print(f"{'='*70}")
    
    bug_dir = os.path.join('multi_bug_data', bug_name)
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
        
        # Create matrix
        code_units = sorted(line_coverage.keys())
        num_tests = len(test_names)
        num_units = len(code_units)
        
        matrix = np.zeros((num_tests, num_units), dtype=int)
        for j, unit in enumerate(code_units):
            if line_coverage[unit]['hits'] > 0:
                matrix[:, j] = 1
        
        # Create labels
        import re
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
        
        test_labels = np.array(test_labels)
        failing_count = test_labels.sum()
        
        print(f"  ✓ {num_tests} tests ({failing_count} failing), {num_units} code lines")
        
        # Sample for visualization
        max_samples = 500
        max_features = 500
        
        if num_tests > max_samples:
            sample_indices = np.random.choice(num_tests, max_samples, replace=False)
            matrix_sample = matrix[sample_indices, :min(max_features, num_units)]
            labels_sample = test_labels[sample_indices]
        else:
            matrix_sample = matrix[:, :min(max_features, num_units)]
            labels_sample = test_labels
        
        # Separate failing and passing
        failing_indices = np.where(labels_sample == 1)[0]
        passing_indices = np.where(labels_sample == 0)[0]
        
        # Create heatmap
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        # Failing tests
        if len(failing_indices) > 0:
            failing_matrix = matrix_sample[failing_indices, :]
            sns.heatmap(failing_matrix, cmap='Reds', cbar=True, 
                       xticklabels=False, yticklabels=False,
                       ax=axes[0], rasterized=True)
            axes[0].set_title(f'{bug_name} - Failing Tests ({len(failing_indices)} tests)', 
                            fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Failing Tests', fontsize=12)
        else:
            axes[0].text(0.5, 0.5, 'No failing tests', ha='center', va='center', fontsize=14)
            axes[0].set_title(f'{bug_name} - Failing Tests (0 tests)', fontsize=14, fontweight='bold')
        
        # Passing tests (sample)
        passing_sample_size = min(200, len(passing_indices))
        if passing_sample_size > 0:
            passing_sample = np.random.choice(passing_indices, passing_sample_size, replace=False)
            passing_matrix = matrix_sample[passing_sample, :]
            sns.heatmap(passing_matrix, cmap='Blues', cbar=True,
                       xticklabels=False, yticklabels=False,
                       ax=axes[1], rasterized=True)
            axes[1].set_title(f'{bug_name} - Passing Tests (sample: {passing_sample_size} of {len(passing_indices)})', 
                            fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Code Lines', fontsize=12)
            axes[1].set_ylabel('Passing Tests', fontsize=12)
        
        plt.tight_layout()
        filename = os.path.join(OUTPUT_DIR, f'heatmap_{bug_name}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: visuals/heatmap_{bug_name}.png")
        created_heatmaps.append(f'heatmap_{bug_name}.png')
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Create combined heatmap
print(f"\n{'='*70}")
print("Creating Combined Heatmap")
print(f"{'='*70}")

try:
    X_combined = np.load('combined/combined_coverage_matrix.npy')
    y_combined = np.load('combined/combined_test_labels.npy')
    
    print(f"  Combined data: {X_combined.shape[0]:,} tests, {X_combined.shape[1]:,} code lines")
    print(f"  Failing tests: {y_combined.sum()}")
    
    # Sample for visualization
    sample_size = min(1000, X_combined.shape[0])
    feature_size = min(800, X_combined.shape[1])
    
    sample_indices = np.random.choice(X_combined.shape[0], sample_size, replace=False)
    X_sample = X_combined[sample_indices, :feature_size]
    y_sample = y_combined[sample_indices]
    
    # Separate failing and passing
    failing_indices = np.where(y_sample == 1)[0]
    passing_indices = np.where(y_sample == 0)[0]
    
    # Create combined heatmap
    fig, axes = plt.subplots(2, 1, figsize=(18, 14))
    
    # Failing tests
    if len(failing_indices) > 0:
        failing_matrix = X_sample[failing_indices, :]
        sns.heatmap(failing_matrix, cmap='Reds', cbar=True,
                  xticklabels=False, yticklabels=False,
                  ax=axes[0], rasterized=True)
        axes[0].set_title(f'Combined All Bugs - Failing Tests ({len(failing_indices)} tests)', 
                        fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Failing Tests', fontsize=14)
    else:
        axes[0].text(0.5, 0.5, 'No failing tests in sample', ha='center', va='center', fontsize=14)
    
    # Passing tests (sample)
    passing_sample_size = min(500, len(passing_indices))
    if passing_sample_size > 0:
        passing_sample = np.random.choice(passing_indices, passing_sample_size, replace=False)
        passing_matrix = X_sample[passing_sample, :]
        sns.heatmap(passing_matrix, cmap='Blues', cbar=True,
                  xticklabels=False, yticklabels=False,
                  ax=axes[1], rasterized=True)
        axes[1].set_title(f'Combined All Bugs - Passing Tests (sample: {passing_sample_size} of {len(passing_indices):,})', 
                        fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Code Lines', fontsize=14)
        axes[1].set_ylabel('Passing Tests', fontsize=14)
    
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, 'heatmap_combined_all_bugs.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: visuals/heatmap_combined_all_bugs.png")
    created_heatmaps.append('heatmap_combined_all_bugs.png')
    
except Exception as e:
    print(f"  ✗ Error creating combined heatmap: {str(e)}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*70}")
print("✓ Heatmap generation complete!")
print(f"{'='*70}")
print(f"\nGenerated {len(created_heatmaps)} heatmaps:")
for hm in created_heatmaps:
    print(f"  - visuals/{hm}")

