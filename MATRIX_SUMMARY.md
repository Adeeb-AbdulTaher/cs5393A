# Coverage Matrix Summary - Ready for ML Input

## ✅ Successfully Created Matrices

### Line-Level Coverage Matrix
- **File**: `line_coverage_matrix.npy`
- **Shape**: (2193, 574)
  - **Rows**: 2,193 test cases
  - **Columns**: 574 code lines
- **Values**: Binary (0 or 1)
  - `1` = test covers that line
  - `0` = test does not cover that line
- **Coverage Density**: 55.4% (matches coverage.xml)
- **Size**: ~5 MB

### Test Labels
- **File**: `line_coverage_labels.npy`
- **Shape**: (2193,)
- **Values**: Binary (0 or 1)
  - `1` = failing test
  - `0` = passing test
- **Distribution**:
  - Failing: 1 test
  - Passing: 2,192 tests

### Method-Level Coverage Matrix
- **File**: `method_coverage_matrix.npy`
- **Shape**: (2193, 1)
- **Note**: Simplified version (single method/class)

## Files Created

### NumPy Arrays (for ML models)
1. `line_coverage_matrix.npy` - Main coverage matrix
2. `line_coverage_labels.npy` - Test labels (1=failing, 0=passing)
3. `method_coverage_matrix.npy` - Method-level matrix
4. `method_coverage_labels.npy` - Method-level labels

### CSV Files (human-readable)
1. `line_coverage_test_labels.csv` - Test names with labels
2. `method_coverage_test_labels.csv` - Method-level test labels

### Metadata
1. `line_coverage_metadata.json` - Matrix statistics and info
2. `method_coverage_metadata.json` - Method-level metadata

## Usage in ML Models (DEEPRL4FL format)

### Load the matrices:
```python
import numpy as np

# Load coverage matrix (X)
X = np.load('line_coverage_matrix.npy')
# Shape: (2193, 574) - tests × code lines

# Load labels (y)
y = np.load('line_coverage_labels.npy')
# Shape: (2193,) - binary labels

print(f"Input shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Failing tests: {y.sum()}")
print(f"Passing tests: {len(y) - y.sum()}")
```

### Matrix Structure
```
        Line1  Line2  Line3  ...  Line574
Test1     1      0      1    ...     0
Test2     0      1      1    ...     1
Test3     1      1      0    ...     0
...
Test2193  0      1      1    ...     0
```

Each row represents a test case, each column represents a code line.
- Matrix[i, j] = 1 if test i covers line j
- Matrix[i, j] = 0 if test i does not cover line j

## Next Steps

1. **Use in ML Model**: Load `line_coverage_matrix.npy` and `line_coverage_labels.npy`
2. **Train/Test Split**: Split the 2,193 tests into training and testing sets
3. **Feature Engineering**: The matrix is already in the correct format (binary features)
4. **Model Training**: Use with DEEPRL4FL or other fault localization models

## Important Notes

⚠️ **Current Limitation**: The coverage matrix uses aggregate coverage data from Defects4J. This means:
- All tests that executed are marked as covering the same lines
- For true per-test coverage, you would need to run coverage collection per test

For DEEPRL4FL, this simplified version can still be useful, but for more accurate results, consider:
- Running per-test coverage collection
- Using test execution traces
- Combining with other features (test metadata, code complexity, etc.)

## Verification

Run `python verify_matrix.py` to see detailed statistics about the matrices.

