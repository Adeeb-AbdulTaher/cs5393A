# Combined Multi-Bug Data Analysis

This folder contains scripts and outputs for analyzing combined data from multiple Defects4J bugs.

## Quick Start

Run these scripts in order:

```powershell
# 1. Combine data from multiple bugs
python combined/combine_multi_bug_data.py

# 2. Generate graphs and visualizations
python combined/generate_graphs.py

# 3. Generate tables
python combined/generate_tables.py

# 4. Generate comprehensive report
python combined/generate_report.py
```

## Scripts

### `combine_multi_bug_data.py`
- Combines coverage data from multiple bugs in `multi_bug_data/`
- Creates combined coverage matrix and labels
- Outputs: `.npy` files, CSV files, metadata JSON

### `generate_graphs.py`
- Creates visualizations (heatmaps, distributions, statistics)
- Outputs: PNG files in `visuals/` folder

### `generate_tables.py`
- Generates analysis tables
- Outputs: CSV files in `tables/` folder

### `generate_report.py`
- Creates comprehensive markdown report
- Outputs: `COMBINED_REPORT.md`

## Output Structure

```
combined/
├── combined_coverage_matrix.npy      # Coverage matrix for ML
├── combined_test_labels.npy          # Labels for ML
├── combined_test_labels.csv          # Human-readable labels
├── combined_bug_info.csv             # Per-bug statistics
├── combined_metadata.json            # Complete metadata
├── visuals/                          # All graphs
│   ├── combined_coverage_heatmap_full.png
│   ├── combined_coverage_heatmap_failing_vs_passing.png
│   ├── combined_coverage_distribution.png
│   ├── combined_bug_statistics.png
│   ├── combined_class_balance.png
│   └── combined_coverage_density.png
├── tables/                           # All tables
│   ├── table_summary_statistics.csv
│   ├── table_bug_analysis.csv
│   ├── table_coverage_statistics.csv
│   ├── table_class_imbalance.csv
│   └── table_before_after_comparison.csv
└── COMBINED_REPORT.md               # Comprehensive report
```

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Notes

- Make sure `multi_bug_data/` folder exists with bug subdirectories
- Each bug subdirectory should contain: `coverage.xml`, `all_tests`, `failing_tests`, `summary.csv`
- Scripts will create output directories automatically

