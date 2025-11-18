# Visualization Summary - Paper-Style Figures Created

## ✅ Successfully Created Visualizations

### 1. **Coverage Matrix Heatmaps**

#### `coverage_heatmap_full.png`
- **Type**: Full coverage matrix heatmap
- **Size**: 2,193 tests × 574 code lines
- **Features**: 
  - Grayscale heatmap showing coverage patterns
  - Red dashed line indicating failing test (if present)
  - Colorbar showing coverage values
- **Use**: Main figure showing overall coverage structure

#### `coverage_heatmap_subset.png`
- **Type**: Subset heatmap (first 100 tests)
- **Size**: 100 tests × 574 code lines
- **Features**:
  - Easier to see individual patterns
  - Red lines highlight failing tests
  - More detailed view of coverage relationships
- **Use**: Detailed view for paper figures

### 2. **Dot Plot Visualization**

#### `coverage_dot_plot.png`
- **Type**: Dot-based coverage plot
- **Size**: 50×50 subset
- **Features**:
  - ● (filled black dots) = Covered lines
  - ○ (hollow white dots) = Not covered lines
  - Red dashed lines = Failing tests
  - Grid for easy reading
- **Use**: Similar to Figure 2 in research papers

### 3. **Coverage Density Analysis**

#### `coverage_density.png` (if created)
- **Type**: Dual-panel density analysis
- **Left Panel**: Coverage density per code line
  - Shows how many tests cover each line
  - Mean line indicated
- **Right Panel**: Coverage density per test
  - Shows how many lines each test covers
  - Red dots = Failing tests
  - Blue dots = Passing tests
- **Use**: Statistical analysis of coverage patterns

### 4. **Failing Test Comparison**

#### `failing_test_comparison.png` (if created)
- **Type**: Side-by-side comparison
- **Top Panel**: Failing test coverage pattern
- **Bottom Panel**: Average passing test coverage pattern
- **Use**: Visual comparison to identify differences

## How to Use These Visualizations

### For Your Paper/Presentation:

1. **Main Figure**: Use `coverage_heatmap_full.png` or `coverage_heatmap_subset.png`
   - Shows the overall structure
   - Demonstrates coverage relationships
   - Professional appearance

2. **Detail View**: Use `coverage_dot_plot.png`
   - Shows individual coverage points
   - Easy to understand format
   - Good for explaining the concept

3. **Analysis**: Use `coverage_density.png`
   - Statistical insights
   - Shows coverage distribution
   - Highlights differences

4. **Comparison**: Use `failing_test_comparison.png`
   - Shows why failing tests are different
   - Visual evidence of patterns

## Customization Tips

### Change Colors:
```python
# In the script, modify cmap parameter:
plt.imshow(X, cmap='viridis')  # or 'plasma', 'inferno', 'coolwarm'
```

### Adjust Size:
```python
plt.figure(figsize=(width, height))  # Change dimensions
```

### Highlight Specific Regions:
```python
# Add vertical lines for buggy code
plt.axvline(x=buggy_line, color='red', linewidth=2)
```

## File Locations

All PNG files are saved in your project directory:
`D:\adeeb\Downloads\Shibbir Presentation\project\`

## Next Steps

1. Review the generated images
2. Select the best ones for your paper
3. Add captions and labels as needed
4. Include in your presentation/report

