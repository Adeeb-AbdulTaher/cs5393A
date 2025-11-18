# Visualization Guide - Paper-Style Figures

## Generated Visualizations

### 1. Coverage Matrix Heatmaps

**File**: `coverage_heatmap_full.png`
- Full coverage matrix (2,193 tests × 574 lines)
- Grayscale heatmap showing coverage patterns
- Red line indicates failing test (if present)

**File**: `coverage_heatmap_subset.png`
- Subset view (first 100 tests)
- Easier to see patterns
- Highlighted failing test rows

### 2. Dot Plot

**File**: `coverage_dot_plot.png`
- Dot-based visualization (50×50 subset)
- ● (filled dot) = Covered line
- ○ (hollow dot) = Not covered
- Red line = Failing test

### 3. Coverage Density Analysis

**File**: `coverage_density.png`
- Left: Coverage density per code line (how many tests cover each line)
- Right: Coverage density per test (how many lines each test covers)
- Red dots = Failing tests

### 4. Failing Test Comparison

**File**: `failing_test_comparison.png` (if failing test exists)
- Top: Coverage pattern of failing test
- Bottom: Average coverage pattern of passing tests
- Visual comparison to identify differences

## How to Use

### Run the visualization script:

```powershell
python create_paper_visualizations.py
```

### Or use individual commands:

```python
# Simple heatmap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X = np.load('line_coverage_matrix.npy')
plt.figure(figsize=(12, 8))
plt.imshow(X, cmap='Greys', aspect='auto')
plt.title('Coverage Matrix')
plt.xlabel('Code Lines')
plt.ylabel('Test Cases')
plt.savefig('my_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
```

## Customization

### Change colors:
```python
# Use different colormap
plt.imshow(X, cmap='viridis')  # or 'plasma', 'inferno', 'magma'
```

### Highlight specific regions:
```python
# Highlight failing test
failing_idx = 100  # your failing test index
plt.axhline(y=failing_idx, color='red', linewidth=2, linestyle='--')

# Highlight specific code lines
plt.axvline(x=buggy_line, color='red', linewidth=2, linestyle='--')
```

### Adjust size:
```python
plt.figure(figsize=(width, height))  # Adjust as needed
```

## For Paper Submission

- Use high DPI (300) for publication quality
- Save as PNG or PDF for best quality
- Include colorbar labels and axis labels
- Use consistent styling across all figures

