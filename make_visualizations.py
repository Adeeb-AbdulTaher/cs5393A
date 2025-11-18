"""
Create paper-style visualizations - Simple and direct version
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Creating visualizations...")

# Load data
X = np.load('line_coverage_matrix.npy')
y = np.load('line_coverage_labels.npy')

print(f"Matrix: {X.shape}, Failing: {y.sum()}")

# 1. Full heatmap
print("1. Full heatmap...")
plt.figure(figsize=(14, 10))
plt.imshow(X, cmap='Greys', aspect='auto', interpolation='nearest')
plt.colorbar(label='Coverage')
plt.title('Coverage Matrix Heatmap\n(Rows: Tests, Columns: Code Lines)', fontsize=14, fontweight='bold')
plt.xlabel('Code Lines (574)', fontsize=12)
plt.ylabel('Test Cases (2,193)', fontsize=12)
failing_idx = np.where(y == 1)[0]
if len(failing_idx) > 0:
    plt.axhline(y=failing_idx[0], color='red', linewidth=2, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('coverage_heatmap_full.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: coverage_heatmap_full.png")

# 2. Subset heatmap
print("2. Subset heatmap...")
subset = 100
X_sub = X[:subset, :]
y_sub = y[:subset]
plt.figure(figsize=(14, 8))
plt.imshow(X_sub, cmap='Greys', aspect='auto', interpolation='nearest')
plt.colorbar(label='Coverage')
plt.title(f'Coverage Matrix (First {subset} Tests)', fontsize=14, fontweight='bold')
plt.xlabel('Code Lines', fontsize=12)
plt.ylabel('Test Cases', fontsize=12)
failing_sub = np.where(y_sub == 1)[0]
for idx in failing_sub:
    plt.axhline(y=idx, color='red', linewidth=2.5, linestyle='--', alpha=0.8)
plt.tight_layout()
plt.savefig('coverage_heatmap_subset.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: coverage_heatmap_subset.png")

# 3. Dot plot
print("3. Dot plot...")
size = 50
X_dot = X[:size, :size]
y_dot = y[:size]
fig, ax = plt.subplots(figsize=(12, 10))
for i in range(size):
    for j in range(size):
        if X_dot[i, j] == 1:
            ax.plot(j, i, 'ko', markersize=4, alpha=0.7)
        else:
            ax.plot(j, i, 'wo', markersize=3, alpha=0.4, markeredgecolor='gray', markeredgewidth=0.5)
failing_dot = np.where(y_dot == 1)[0]
for idx in failing_dot:
    ax.axhline(y=idx, color='red', linewidth=2, alpha=0.6, linestyle='--')
ax.set_title(f'Coverage Dot Plot ({size}×{size})\n● = Covered, ○ = Not Covered', fontsize=14, fontweight='bold')
ax.set_xlabel('Code Lines', fontsize=12)
ax.set_ylabel('Test Cases', fontsize=12)
ax.grid(True, alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('coverage_dot_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: coverage_dot_plot.png")

# 4. Density plots
print("4. Density plots...")
line_density = X.sum(axis=0)
test_density = X.sum(axis=1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.plot(line_density, 'b-', alpha=0.7, linewidth=0.5)
ax1.axhline(y=line_density.mean(), color='r', linestyle='--', label=f'Mean: {line_density.mean():.1f}')
ax1.fill_between(range(len(line_density)), 0, line_density, alpha=0.3)
ax1.set_title('Coverage per Code Line', fontsize=12, fontweight='bold')
ax1.set_xlabel('Line Index')
ax1.set_ylabel('Tests Covering')
ax1.legend()
ax1.grid(True, alpha=0.3)
colors = ['red' if y[i] == 1 else 'blue' for i in range(len(test_density))]
ax2.scatter(range(len(test_density)), test_density, c=colors, alpha=0.5, s=10)
ax2.axhline(y=test_density.mean(), color='g', linestyle='--', label=f'Mean: {test_density.mean():.1f}')
ax2.set_title('Coverage per Test (Red=Failing)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Test Index')
ax2.set_ylabel('Lines Covered')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('coverage_density.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: coverage_density.png")

# 5. Failing test comparison
if len(failing_idx) > 0:
    print("5. Failing test comparison...")
    failing_coverage = X[failing_idx[0], :]
    passing_indices = np.where(y == 0)[0]
    avg_passing = X[passing_indices, :].mean(axis=0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    ax1.bar(range(len(failing_coverage)), failing_coverage, color='red', alpha=0.7, width=1.0)
    ax1.set_title(f'Failing Test Coverage (Test #{failing_idx[0]})', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Coverage')
    ax1.set_ylim([-0.1, 1.1])
    ax1.grid(True, alpha=0.3, axis='y')
    ax2.bar(range(len(avg_passing)), avg_passing, color='steelblue', alpha=0.7, width=1.0)
    ax2.set_title('Average Passing Test Coverage', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Code Line Index')
    ax2.set_ylabel('Average Coverage')
    ax2.set_ylim([-0.1, 1.1])
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('failing_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: failing_test_comparison.png")

print("\n✓ All visualizations created!")

