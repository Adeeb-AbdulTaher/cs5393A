"""
Create visualizations of coverage matrices similar to research paper figures
- Coverage matrix heatmaps
- Dot plots for test coverage
- Highlighting buggy locations
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("Coverage Matrix Visualizations")
print("=" * 60)

# Load data
print("\n1. Loading data...")
X = np.load('line_coverage_matrix.npy')
y = np.load('line_coverage_labels.npy')

# Load test names for labeling
test_df = pd.read_csv('line_coverage_test_labels.csv')
test_names = test_df['test_name'].values

print(f"   Coverage matrix shape: {X.shape}")
print(f"   Labels shape: {y.shape}")
print(f"   Failing tests: {y.sum()}")

# Find failing test indices
failing_indices = np.where(y == 1)[0]
print(f"   Failing test indices: {failing_indices}")

# ============================================================================
# A. CODE COVERAGE HEATMAP (Full Matrix)
# ============================================================================
print("\n2. Creating full coverage matrix heatmap...")
plt.figure(figsize=(14, 10))
sns.heatmap(X, cmap="Greys", cbar=True, cbar_kws={'label': 'Coverage (1=covered, 0=not covered)'},
            xticklabels=False, yticklabels=False, rasterized=True)
plt.title("Code Coverage Matrix Heatmap\n(Rows: Test Cases, Columns: Code Lines)", 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Code Lines (574 total)", fontsize=12)
plt.ylabel("Test Cases (2,193 total)", fontsize=12)
plt.tight_layout()
plt.savefig('coverage_heatmap_full.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: coverage_heatmap_full.png")
plt.close()

# ============================================================================
# B. CODE COVERAGE HEATMAP (Subset - First 100 tests, all lines)
# ============================================================================
print("\n3. Creating subset heatmap (first 100 tests)...")
subset_size = 100
X_subset = X[:subset_size, :]
y_subset = y[:subset_size]

plt.figure(figsize=(14, 8))
sns.heatmap(X_subset, cmap="Greys", cbar=True, 
            cbar_kws={'label': 'Coverage'},
            xticklabels=False, yticklabels=50,  # Show every 50th test label
            rasterized=True)

# Highlight failing test rows
for idx in np.where(y_subset == 1)[0]:
    plt.axhline(y=idx, color='red', linewidth=2, alpha=0.7, linestyle='--')

plt.title(f"Coverage Matrix Heatmap (Subset: First {subset_size} Tests)\nRed lines indicate failing tests", 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Code Lines (574 total)", fontsize=12)
plt.ylabel("Test Cases", fontsize=12)
plt.tight_layout()
plt.savefig('coverage_heatmap_subset.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: coverage_heatmap_subset.png")
plt.close()

# ============================================================================
# C. DOT PLOT (Small subset for clarity)
# ============================================================================
print("\n4. Creating dot plot (first 50 tests, first 50 lines)...")
dot_size = 50
X_dot = X[:dot_size, :dot_size]
y_dot = y[:dot_size]

plt.figure(figsize=(12, 10))
for i in range(dot_size):
    for j in range(dot_size):
        val = X_dot[i, j]
        if val == 1:
            plt.plot(j, i, 'ko', markersize=3, alpha=0.6)  # Filled dot for covered
        else:
            plt.plot(j, i, 'wo', markersize=2, alpha=0.3, markeredgecolor='gray', markeredgewidth=0.5)  # Hollow dot for not covered

# Highlight failing test rows
for idx in np.where(y_dot == 1)[0]:
    plt.axhline(y=idx, color='red', linewidth=2, alpha=0.5, linestyle='--', label='Failing test' if idx == np.where(y_dot == 1)[0][0] else '')

plt.title(f"Code Coverage Dot Plot\n(Subset: {dot_size} Tests × {dot_size} Lines)\n● = Covered, ○ = Not Covered", 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Code Lines", fontsize=12)
plt.ylabel("Test Cases", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('coverage_dot_plot.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: coverage_dot_plot.png")
plt.close()

# ============================================================================
# D. COVERAGE DENSITY HEATMAP (Aggregated view)
# ============================================================================
print("\n5. Creating coverage density heatmap...")
# Calculate coverage density per line (how many tests cover each line)
line_coverage_density = X.sum(axis=0)  # Sum across tests for each line

# Calculate test coverage density (how many lines each test covers)
test_coverage_density = X.sum(axis=1)  # Sum across lines for each test

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Line coverage density
ax1.bar(range(len(line_coverage_density)), line_coverage_density, color='steelblue', alpha=0.7)
ax1.axhline(y=line_coverage_density.mean(), color='red', linestyle='--', 
            label=f'Mean: {line_coverage_density.mean():.1f}')
ax1.set_title('Coverage Density per Code Line\n(How many tests cover each line)', 
              fontsize=12, fontweight='bold')
ax1.set_xlabel('Code Line Index', fontsize=11)
ax1.set_ylabel('Number of Tests Covering Line', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Test coverage density
colors = ['red' if y[i] == 1 else 'steelblue' for i in range(len(test_coverage_density))]
ax2.bar(range(len(test_coverage_density)), test_coverage_density, color=colors, alpha=0.7)
ax2.axhline(y=test_coverage_density.mean(), color='green', linestyle='--', 
            label=f'Mean: {test_coverage_density.mean():.1f}')
ax2.set_title('Coverage Density per Test\n(How many lines each test covers)', 
              fontsize=12, fontweight='bold')
ax2.set_xlabel('Test Case Index', fontsize=11)
ax2.set_ylabel('Number of Lines Covered', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coverage_density.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: coverage_density.png")
plt.close()

# ============================================================================
# E. FAILING TEST FOCUSED VIEW
# ============================================================================
print("\n6. Creating failing test focused visualization...")
if len(failing_indices) > 0:
    failing_idx = failing_indices[0]
    
    # Get coverage pattern for failing test
    failing_test_coverage = X[failing_idx, :]
    
    # Compare with average passing test coverage
    passing_indices = np.where(y == 0)[0]
    avg_passing_coverage = X[passing_indices, :].mean(axis=0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Failing test coverage
    ax1.bar(range(len(failing_test_coverage)), failing_test_coverage, 
            color='red', alpha=0.7, width=1.0)
    ax1.set_title(f'Failing Test Coverage Pattern\n(Test index: {failing_idx})', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Code Line Index', fontsize=11)
    ax1.set_ylabel('Coverage (1=covered, 0=not covered)', fontsize=11)
    ax1.set_ylim([-0.1, 1.1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Average passing test coverage
    ax2.bar(range(len(avg_passing_coverage)), avg_passing_coverage, 
            color='steelblue', alpha=0.7, width=1.0)
    ax2.set_title('Average Passing Test Coverage Pattern', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Code Line Index', fontsize=11)
    ax2.set_ylabel('Average Coverage', fontsize=11)
    ax2.set_ylim([-0.1, 1.1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('failing_test_coverage.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: failing_test_coverage.png")
    plt.close()

# ============================================================================
# F. STATISTICS SUMMARY
# ============================================================================
print("\n7. Coverage Statistics:")
print("-" * 60)
print(f"   Total tests: {len(y)}")
print(f"   Total code lines: {X.shape[1]}")
print(f"   Failing tests: {y.sum()}")
print(f"   Passing tests: {len(y) - y.sum()}")
print(f"   Overall coverage: {X.mean() * 100:.2f}%")
print(f"   Average lines covered per test: {X.sum(axis=1).mean():.1f}")
print(f"   Average tests covering each line: {X.sum(axis=0).mean():.1f}")
print(f"   Most covered line: {X.sum(axis=0).argmax()} ({X.sum(axis=0).max()} tests)")
print(f"   Least covered line: {X.sum(axis=0).argmin()} ({X.sum(axis=0).min()} tests)")

print("\n" + "=" * 60)
print("✓ All visualizations created successfully!")
print("=" * 60)
print("\nGenerated files:")
print("  - coverage_heatmap_full.png: Full matrix heatmap")
print("  - coverage_heatmap_subset.png: Subset heatmap with failing test highlights")
print("  - coverage_dot_plot.png: Dot plot visualization")
print("  - coverage_density.png: Coverage density analysis")
print("  - failing_test_coverage.png: Failing test focused view")

