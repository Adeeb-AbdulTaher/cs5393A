"""
Generate visualizations for combined multi-bug data
Creates heatmaps, dot plots, and coverage analysis graphs
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR
VISUALS_DIR = os.path.join(OUTPUT_DIR, 'visuals')
os.makedirs(VISUALS_DIR, exist_ok=True)

print("=" * 60)
print("Generating Graphs for Combined Data")
print("=" * 60)

# Load data
print("\nLoading data...")
try:
    X = np.load(os.path.join(OUTPUT_DIR, 'combined_coverage_matrix.npy'))
    y = np.load(os.path.join(OUTPUT_DIR, 'combined_test_labels.npy'))
    bug_info = pd.read_csv(os.path.join(OUTPUT_DIR, 'combined_bug_info.csv'))
    metadata = json.load(open(os.path.join(OUTPUT_DIR, 'combined_metadata.json')))
    print(f"  ✓ Loaded matrix: {X.shape}")
    print(f"  ✓ Loaded labels: {len(y)} tests ({y.sum()} failing)")
except Exception as e:
    print(f"  ✗ Error loading data: {e}")
    print("  Make sure to run combine_multi_bug_data.py first!")
    exit(1)

# 1. Coverage Matrix Heatmap (Full)
print("\n[1/6] Creating full coverage heatmap...")
plt.figure(figsize=(16, 12))
# Sample for visualization (too large to show all)
sample_size = min(500, X.shape[0])
sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
X_sample = X[sample_indices, :min(500, X.shape[1])]
y_sample = y[sample_indices]

# Create heatmap
sns.heatmap(X_sample, cmap='YlOrRd', cbar=True, 
            xticklabels=False, yticklabels=False,
            rasterized=True)
plt.title(f'Coverage Matrix Heatmap (Sample: {sample_size} tests × {X_sample.shape[1]} lines)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Code Lines', fontsize=12)
plt.ylabel('Test Cases', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'combined_coverage_heatmap_full.png'), 
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: visuals/combined_coverage_heatmap_full.png")

# 2. Coverage Matrix Heatmap (Failing vs Passing)
print("\n[2/6] Creating failing vs passing heatmap...")
failing_indices = np.where(y == 1)[0]
passing_indices = np.where(y == 0)[0]

# Sample failing and passing tests
n_failing = min(50, len(failing_indices))
n_passing = min(200, len(passing_indices))

if n_failing > 0 and n_passing > 0:
    failing_sample = np.random.choice(failing_indices, n_failing, replace=False)
    passing_sample = np.random.choice(passing_indices, n_passing, replace=False)
    
    combined_indices = np.concatenate([failing_sample, passing_sample])
    X_compare = X[combined_indices, :min(300, X.shape[1])]
    y_compare = y[combined_indices]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(X_compare, cmap='YlOrRd', cbar=True,
                xticklabels=False, yticklabels=False,
                rasterized=True)
    plt.title(f'Coverage Matrix: Failing (top {n_failing}) vs Passing (bottom {n_passing})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Code Lines', fontsize=12)
    plt.ylabel('Test Cases', fontsize=12)
    plt.axhline(y=n_failing, color='blue', linestyle='--', linewidth=2, label='Failing/Passing boundary')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'combined_coverage_heatmap_failing_vs_passing.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: visuals/combined_coverage_heatmap_failing_vs_passing.png")

# 3. Dot Plot - Coverage per Test
print("\n[3/6] Creating coverage per test plot...")
coverage_per_test = X.sum(axis=1)
failing_coverage = coverage_per_test[y == 1]
passing_coverage = coverage_per_test[y == 0]

plt.figure(figsize=(12, 8))
plt.hist(passing_coverage, bins=50, alpha=0.7, label=f'Passing ({len(passing_coverage)})', color='green')
plt.hist(failing_coverage, bins=50, alpha=0.7, label=f'Failing ({len(failing_coverage)})', color='red')
plt.xlabel('Lines Covered per Test', fontsize=12)
plt.ylabel('Number of Tests', fontsize=12)
plt.title('Coverage Distribution: Failing vs Passing Tests', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'combined_coverage_distribution.png'), 
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: visuals/combined_coverage_distribution.png")

# 4. Bug Statistics Bar Chart
print("\n[4/6] Creating bug statistics chart...")

# Calculate passing if not present
if 'passing' not in bug_info.columns:
    bug_info['passing'] = bug_info['tests'] - bug_info['failing']

# Fill missing bug names
if 'bug' in bug_info.columns:
    bug_info['bug'] = bug_info['bug'].fillna('')
    if bug_info['bug'].str.strip().eq('').all():
        bug_info['bug'] = [f"Chart_{i+1}" for i in range(len(bug_info))]
else:
    bug_info['bug'] = [f"Chart_{i+1}" for i in range(len(bug_info))]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Tests per bug
axes[0, 0].bar(bug_info['bug'], bug_info['tests'], color='steelblue')
axes[0, 0].set_title('Total Tests per Bug', fontweight='bold')
axes[0, 0].set_xlabel('Bug')
axes[0, 0].set_ylabel('Number of Tests')
axes[0, 0].tick_params(axis='x', rotation=45)

# Failing tests per bug
axes[0, 1].bar(bug_info['bug'], bug_info['failing'], color='crimson')
axes[0, 1].set_title('Failing Tests per Bug', fontweight='bold')
axes[0, 1].set_xlabel('Bug')
axes[0, 1].set_ylabel('Number of Failing Tests')
axes[0, 1].tick_params(axis='x', rotation=45)

# Passing tests per bug
axes[1, 0].bar(bug_info['bug'], bug_info['passing'], color='green')
axes[1, 0].set_title('Passing Tests per Bug', fontweight='bold')
axes[1, 0].set_xlabel('Bug')
axes[1, 0].set_ylabel('Number of Passing Tests')
axes[1, 0].tick_params(axis='x', rotation=45)

# Code lines per bug
axes[1, 1].bar(bug_info['bug'], bug_info['lines'], color='orange')
axes[1, 1].set_title('Code Lines per Bug', fontweight='bold')
axes[1, 1].set_xlabel('Bug')
axes[1, 1].set_ylabel('Number of Code Lines')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'combined_bug_statistics.png'), 
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: visuals/combined_bug_statistics.png")

# 5. Class Balance Visualization
print("\n[5/6] Creating class balance visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
failing_count = int(y.sum())
passing_count = int(len(y) - y.sum())
axes[0].pie([failing_count, passing_count], 
            labels=[f'Failing\n({failing_count})', f'Passing\n({passing_count})'],
            colors=['red', 'green'], autopct='%1.2f%%', startangle=90)
axes[0].set_title('Test Class Distribution', fontweight='bold', fontsize=12)

# Bar chart
axes[1].bar(['Failing', 'Passing'], [failing_count, passing_count], 
            color=['red', 'green'])
axes[1].set_title('Test Counts by Class', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Number of Tests')
for i, v in enumerate([failing_count, passing_count]):
    axes[1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'combined_class_balance.png'), 
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: visuals/combined_class_balance.png")

# 6. Coverage Density Plot
print("\n[6/6] Creating coverage density plot...")
plt.figure(figsize=(12, 8))
plt.hist(coverage_per_test, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
plt.axvline(coverage_per_test.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {coverage_per_test.mean():.1f}')
plt.axvline(np.median(coverage_per_test), color='green', linestyle='--', linewidth=2, 
            label=f'Median: {np.median(coverage_per_test):.1f}')
plt.xlabel('Lines Covered per Test', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Coverage Density Distribution (All Tests)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'combined_coverage_density.png'), 
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: visuals/combined_coverage_density.png")

print(f"\n{'='*60}")
print("✓ All graphs generated successfully!")
print(f"{'='*60}")
print(f"\nGraphs saved to: {VISUALS_DIR}/")
print("  - combined_coverage_heatmap_full.png")
print("  - combined_coverage_heatmap_failing_vs_passing.png")
print("  - combined_coverage_distribution.png")
print("  - combined_bug_statistics.png")
print("  - combined_class_balance.png")
print("  - combined_coverage_density.png")

