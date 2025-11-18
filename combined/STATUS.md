# Combined Data Analysis - Status Report

## ✅ Completed

### Data Files
- ✅ `combined_coverage_matrix.npy` - Coverage matrix (10,783 × 1,808)
- ✅ `combined_test_labels.npy` - Test labels (10,783 labels)
- ✅ `combined_test_labels.csv` - Human-readable labels
- ✅ `combined_bug_info.csv` - Per-bug statistics
- ✅ `combined_metadata.json` - Complete metadata

### Tables Generated (in `tables/` folder)
- ✅ `table_summary_statistics.csv` - Overall summary
- ✅ `table_bug_analysis.csv` - Bug-by-bug analysis
- ✅ `table_coverage_statistics.csv` - Coverage statistics
- ✅ `table_class_imbalance.csv` - Class imbalance metrics
- ✅ `table_before_after_comparison.csv` - Before/after comparison

### Scripts
- ✅ `combine_multi_bug_data.py` - Data combination (completed)
- ✅ `generate_tables.py` - Table generation (completed)
- ⚠️ `generate_graphs.py` - Graph generation (needs debugging)
- ⚠️ `generate_report.py` - Report generation (needs debugging)

## 📊 Key Results

### Dataset Statistics
- **Total Bugs**: 5 (Chart-1 through Chart-5)
- **Total Tests**: 10,783
- **Failing Tests**: 27 (0.25%)
- **Passing Tests**: 10,756 (99.75%)
- **Code Units**: 1,808 lines
- **Matrix Size**: 10,783 × 1,808

### Class Balance Improvement
- **Before (Single Bug)**: 1 failing / 2,193 total (0.046%)
- **After (Combined)**: 27 failing / 10,783 total (0.25%)
- **Improvement**: **27x more failing tests!**

### Per-Bug Breakdown
| Bug | Tests | Failing | Passing | Code Lines |
|-----|-------|---------|---------|------------|
| Chart_1 | 2,193 | 1 | 2,192 | 574 |
| Chart_2 | 2,191 | 2 | 2,189 | 801 |
| Chart_3 | 2,187 | 1 | 2,186 | 380 |
| Chart_4 | 2,179 | 22 | 2,157 | 1,808 |
| Chart_5 | 2,033 | 1 | 2,032 | 172 |

## ⚠️ Pending

### Graphs (in `visuals/` folder)
The graph generation script may need debugging. Expected outputs:
- `combined_coverage_heatmap_full.png`
- `combined_coverage_heatmap_failing_vs_passing.png`
- `combined_coverage_distribution.png`
- `combined_bug_statistics.png`
- `combined_class_balance.png`
- `combined_coverage_density.png`

### Report
- `COMBINED_REPORT.md` - Comprehensive markdown report

## 🚀 Next Steps

1. Debug graph generation (check matplotlib backend, dependencies)
2. Generate report manually if script fails
3. Use combined data for ML training:
   ```python
   X = np.load('combined/combined_coverage_matrix.npy')
   y = np.load('combined/combined_test_labels.npy')
   ```

## 📁 File Structure

```
combined/
├── combine_multi_bug_data.py
├── generate_graphs.py
├── generate_tables.py
├── generate_report.py
├── run_all.py
├── README.md
├── STATUS.md (this file)
├── combined_coverage_matrix.npy
├── combined_test_labels.npy
├── combined_test_labels.csv
├── combined_bug_info.csv
├── combined_metadata.json
├── tables/
│   ├── table_summary_statistics.csv
│   ├── table_bug_analysis.csv
│   ├── table_coverage_statistics.csv
│   ├── table_class_imbalance.csv
│   └── table_before_after_comparison.csv
└── visuals/ (empty - needs debugging)
```

