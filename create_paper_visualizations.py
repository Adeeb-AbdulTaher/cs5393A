"""
Create paper-style visualizations for coverage matrices
Similar to DEEPRL4FL and fault localization research papers
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("Creating paper-style visualizations...")
print("=" * 60)

# Load data
X = np.load('line_coverage_matrix.npy')
y = np.load('line_coverage_labels.npy')
test_df = pd.read_csv('line_coverage_test_labels.csv')

print(f"Loaded: {X.shape[0]} tests × {X.shape[1]} lines")
print(f"Failing tests: {y.sum()}")

failing_idx = np.where(y == 1)[0][0] if y.sum() > 0 else None

# ============================================================================
# 1. FULL COVERAGE MATRIX HEATMAP
# ============================================================================
print("\n1. Creating full coverage heatmap...")
plt.figure(figsize=(16, 12))
plt.imshow(X, cmap='Greys', aspect='auto', interpolation='nearest')
plt.colorbar(label='Coverage (1=covered, 0=not covered)')
plt.title('Code Coverage Matrix\n(Rows: Test Cases, Columns: Code Lines)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Code Lines (574 total)', fontsize=14)
plt.ylabel('Test Cases (2,193 total)', fontsize=14)
if failing_idx is not None:
    plt.axhline(y=failing_idx, color='red', linewidth=2, alpha=0.7, 
                linestyle='--', label='Failing test')
    plt.legend()
plt.tight_layout()
plt.savefig('coverage_heatmap_full.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: coverage_heatmap_full.png")

# ============================================================================
# 2. SUBSET HEATMAP (First 100 tests)
# ============================================================================
print("\n2. Creating subset heatmap...")
subset = 100
X_sub = X[:subset, :]
y_sub = y[:subset]

plt.figure(figsize=(14, 8))
plt.imshow(X_sub, cmap='Greys', aspect='auto', interpolation='nearest')
plt.colorbar(label='Coverage')
plt.title(f'Coverage Matrix Heatmap (Subset: First {subset} Tests)\nRed line = Failing test', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Code Lines', fontsize=12)
plt.ylabel('Test Cases', fontsize=12)

failing_sub = np.where(y_sub == 1)[0]
for idx in failing_sub:
    plt.axhline(y=idx, color='red', linewidth=2.5, alpha=0.8, linestyle='--')

plt.tight_layout()
plt.savefig('coverage_heatmap_subset.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: coverage_heatmap_subset.png")

# ============================================================================
# 3. DOT PLOT (Small subset for clarity)
# ============================================================================
print("\n3. Creating dot plot...")
size = 50
X_dot = X[:size, :size]
y_dot = y[:size]

fig, ax = plt.subplots(figsize=(12, 10))
for i in range(size):
    for j in range(size):
        if X_dot[i, j] == 1:
            ax.plot(j, i, 'ko', markersize=4, alpha=0.7)  # Filled dot
        else:
            ax.plot(j, i, 'wo', markersize=3, alpha=0.4, 
                   markeredgecolor='gray', markeredgewidth=0.5)  # Hollow dot

# Highlight failing test
failing_dot = np.where(y_dot == 1)[0]
for idx in failing_dot:
    ax.axhline(y=idx, color='red', linewidth=2, alpha=0.6, linestyle='--')

ax.set_title(f'Coverage Dot Plot ({size}×{size})\n● = Covered, ○ = Not Covered', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Code Lines', fontsize=12)
ax.set_ylabel('Test Cases', fontsize=12)
ax.grid(True, alpha=0.3)
ax.invert_yaxis()  # Match matrix convention
plt.tight_layout()
plt.savefig('coverage_dot_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: coverage_dot_plot.png")

# ============================================================================
# 4. COVERAGE DENSITY ANALYSIS
# ============================================================================
print("\n4. Creating coverage density plots...")
line_density = X.sum(axis=0)
test_density = X.sum(axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Line coverage density
ax1.plot(line_density, 'b-', alpha=0.7, linewidth=0.5)
ax1.axhline(y=line_density.mean(), color='r', linestyle='--', 
           label=f'Mean: {line_density.mean():.1f}')
ax1.fill_between(range(len(line_density)), 0, line_density, alpha=0.3)
ax1.set_title('Coverage Density per Code Line', fontsize=12, fontweight='bold')
ax1.set_xlabel('Code Line Index', fontsize=11)
ax1.set_ylabel('Number of Tests Covering', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Test coverage density
colors = ['red' if y[i] == 1 else 'blue' for i in range(len(test_density))]
ax2.scatter(range(len(test_density)), test_density, c=colors, alpha=0.5, s=10)
ax2.axhline(y=test_density.mean(), color='g', linestyle='--', 
           label=f'Mean: {test_density.mean():.1f}')
ax2.set_title('Coverage Density per Test\n(Red = Failing)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Test Case Index', fontsize=11)
ax2.set_ylabel('Number of Lines Covered', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coverage_density.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: coverage_density.png")

# ============================================================================
# 5. FAILING TEST COMPARISON
# ============================================================================
if failing_idx is not None:
    print("\n5. Creating failing test comparison...")
    failing_coverage = X[failing_idx, :]
    passing_indices = np.where(y == 0)[0]
    avg_passing = X[passing_indices, :].mean(axis=0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Failing test
    ax1.bar(range(len(failing_coverage)), failing_coverage, 
           color='red', alpha=0.7, width=1.0)
    ax1.set_title(f'Failing Test Coverage Pattern (Test #{failing_idx})', 
                 fontsize=12, fontweight='bold')
    ax1.set_ylabel('Coverage', fontsize=11)
    ax1.set_ylim([-0.1, 1.1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Average passing
    ax2.bar(range(len(avg_passing)), avg_passing, 
           color='steelblue', alpha=0.7, width=1.0)
    ax2.set_title('Average Passing Test Coverage Pattern', 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('Code Line Index', fontsize=11)
    ax2.set_ylabel('Average Coverage', fontsize=11)
    ax2.set_ylim([-0.1, 1.1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('failing_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: failing_test_comparison.png")

print("\n" + "=" * 60)
print("✓ All visualizations created!")
print("=" * 60)

