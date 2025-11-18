"""
Generate comprehensive report for combined multi-bug data
Creates a markdown report with all findings, tables, and visualizations
"""
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR

print("=" * 60)
print("Generating Combined Data Report")
print("=" * 60)

# Load data
print("\nLoading data...")
try:
    metadata = json.load(open(os.path.join(OUTPUT_DIR, 'combined_metadata.json')))
    bug_info = pd.read_csv(os.path.join(OUTPUT_DIR, 'combined_bug_info.csv'))
    summary_stats = pd.read_csv(os.path.join(OUTPUT_DIR, 'tables', 'table_summary_statistics.csv'))
    bug_analysis = pd.read_csv(os.path.join(OUTPUT_DIR, 'tables', 'table_bug_analysis.csv'))
    comparison = pd.read_csv(os.path.join(OUTPUT_DIR, 'tables', 'table_before_after_comparison.csv'))
    print(f"  ✓ Loaded metadata and tables")
except Exception as e:
    print(f"  ✗ Error loading data: {e}")
    print("  Make sure to run combine_multi_bug_data.py and generate_tables.py first!")
    exit(1)

# Generate report
report = f"""# Combined Multi-Bug Data Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

This report presents the analysis of combined coverage data from **{metadata['num_bugs']} Defects4J bugs** (Chart project, bugs 1-{metadata['num_bugs']}). The combination of multiple bugs addresses the severe class imbalance present in single-bug datasets, providing a more balanced dataset for machine learning-based fault localization.

### Key Findings

- **Total Tests Collected:** {metadata['total_tests']:,}
- **Failing Tests:** {metadata['total_failing']} ({metadata['failing_percentage']:.2f}%)
- **Passing Tests:** {metadata['total_passing']:,} ({(metadata['total_passing']/metadata['total_tests']*100):.2f}%)
- **Code Units Analyzed:** {metadata['code_units']:,} lines
- **Coverage Matrix Size:** {metadata['matrix_shape'][0]:,} × {metadata['matrix_shape'][1]:,}

---

## 1. Dataset Overview

### 1.1 Bugs Analyzed

The following bugs from the Defects4J Chart project were included in this analysis:

"""

# Add bug details
for i, bug in enumerate(metadata['bugs'], 1):
    report += f"""
**{bug['bug']}**
- Total Tests: {bug['tests']:,}
- Failing Tests: {bug['failing']}
- Passing Tests: {bug['passing']}
- Code Lines: {bug['lines']:,}
"""

report += f"""

### 1.2 Summary Statistics

| Metric | Value |
|--------|-------|
"""

# Add summary statistics
for _, row in summary_stats.iterrows():
    report += f"| {row['Metric']} | {row['Value']} |\n"

report += f"""

---

## 2. Class Imbalance Analysis

### 2.1 Before vs After Comparison

The combination of multiple bugs significantly improves class balance:

| Metric | Before (Single Bug) | After (Combined) | Improvement |
|--------|---------------------|------------------|-------------|
"""

# Add comparison
for _, row in comparison.iterrows():
    report += f"| {row['Metric']} | {row['Before (Single Bug)']} | {row['After (Combined)']} | {row['Improvement']} |\n"

report += f"""

### 2.2 Class Distribution

- **Failing Tests:** {metadata['total_failing']} ({metadata['failing_percentage']:.2f}%)
- **Passing Tests:** {metadata['total_passing']:,} ({(metadata['total_passing']/metadata['total_tests']*100):.2f}%)
- **Imbalance Ratio:** {metadata['total_passing']/metadata['total_failing']:.1f}:1 (Passing:Failing)

**Note:** While class imbalance still exists, it has been significantly reduced from the original 0.046% failure rate to {metadata['failing_percentage']:.2f}%, representing a **{metadata['failing_percentage']/0.046:.1f}x improvement**.

---

## 3. Bug-by-Bug Analysis

### 3.1 Test Distribution per Bug

| Bug | Total Tests | Failing | Passing | Failing % | Passing % | Code Lines |
|-----|-------------|---------|---------|-----------|-----------|------------|
"""

# Add bug analysis
for _, row in bug_analysis.iterrows():
    report += f"| {row['Bug']} | {row['Total Tests']} | {row['Failing']} | {row['Passing']} | {row['Failing %']}% | {row['Passing %']}% | {row['Code Lines']} |\n"

report += f"""

### 3.2 Key Observations

"""

# Find bugs with most/least failing tests
max_failing_bug = bug_analysis.loc[bug_analysis['Failing'].idxmax()]
min_failing_bug = bug_analysis.loc[bug_analysis['Failing'].idxmin()]
avg_failing = bug_analysis['Failing'].mean()

report += f"""
- **Bug with Most Failing Tests:** {max_failing_bug['Bug']} ({max_failing_bug['Failing']} failing tests)
- **Bug with Least Failing Tests:** {min_failing_bug['Bug']} ({min_failing_bug['Failing']} failing tests)
- **Average Failing Tests per Bug:** {avg_failing:.2f}
- **Total Failing Tests Across All Bugs:** {bug_analysis['Failing'].sum()}
"""

report += f"""

---

## 4. Visualizations

The following visualizations provide insights into the combined dataset:

### 4.1 Coverage Matrix Heatmaps

- **Full Coverage Heatmap** (`visuals/combined_coverage_heatmap_full.png`)
  - Shows coverage patterns across all tests and code lines
  - Sample visualization of the {metadata['matrix_shape'][0]:,} × {metadata['matrix_shape'][1]:,} matrix

- **Failing vs Passing Comparison** (`visuals/combined_coverage_heatmap_failing_vs_passing.png`)
  - Side-by-side comparison of coverage patterns
  - Highlights differences between failing and passing tests

### 4.2 Coverage Distribution

- **Coverage Distribution** (`visuals/combined_coverage_distribution.png`)
  - Histogram showing lines covered per test
  - Separate distributions for failing and passing tests

- **Coverage Density** (`visuals/combined_coverage_density.png`)
  - Overall coverage density across all tests
  - Mean and median coverage indicators

### 4.3 Bug Statistics

- **Bug Statistics** (`visuals/combined_bug_statistics.png`)
  - Bar charts showing:
    - Total tests per bug
    - Failing tests per bug
    - Passing tests per bug
    - Code lines per bug

### 4.4 Class Balance

- **Class Balance Visualization** (`visuals/combined_class_balance.png`)
  - Pie chart and bar chart showing failing vs passing test distribution
  - Visual representation of class imbalance

---

## 5. Data Files Generated

### 5.1 NumPy Arrays (for ML Training)

- `combined_coverage_matrix.npy` - Coverage matrix ({metadata['matrix_shape'][0]:,} × {metadata['matrix_shape'][1]:,})
- `combined_test_labels.npy` - Binary labels ({metadata['total_tests']:,} labels)

### 5.2 CSV Files

- `combined_test_labels.csv` - Test names with labels
- `combined_bug_info.csv` - Per-bug statistics
- `tables/table_summary_statistics.csv` - Overall summary
- `tables/table_bug_analysis.csv` - Bug-by-bug analysis
- `tables/table_coverage_statistics.csv` - Coverage statistics
- `tables/table_class_imbalance.csv` - Class imbalance metrics
- `tables/table_before_after_comparison.csv` - Before/after comparison

### 5.3 Metadata

- `combined_metadata.json` - Complete dataset metadata in JSON format

---

## 6. Recommendations

### 6.1 For Machine Learning

1. **Class Weighting**: Use class weights in training to further address imbalance
2. **Stratified Sampling**: Use stratified train/test splits to maintain class distribution
3. **Data Augmentation**: Consider synthetic minority oversampling (SMOTE) if needed
4. **Evaluation Metrics**: Focus on precision, recall, and F1-score rather than accuracy

### 6.2 For Further Data Collection

1. **Expand to More Bugs**: Collect data from all 26 Chart bugs
2. **Multi-Project**: Include bugs from other Defects4J projects (Math, Time, Lang, etc.)
3. **Per-Test Coverage**: Consider using GZoltar for per-test coverage data (more accurate)

---

## 7. Next Steps

1. ✅ **Data Collection** - Completed ({metadata['num_bugs']} bugs)
2. ✅ **Data Combination** - Completed
3. ✅ **Visualization** - Completed
4. ✅ **Table Generation** - Completed
5. ⏭️ **Model Training** - Ready to train with combined data
6. ⏭️ **Evaluation** - Evaluate model performance on combined dataset
7. ⏭️ **Comparison** - Compare results with single-bug baseline

---

## 8. Technical Details

### 8.1 Data Processing

- **Coverage Format**: Cobertura XML (aggregate coverage)
- **Matrix Creation**: Simplified approach (all tests marked as covering all lines with hits > 0)
- **Label Extraction**: Parsed from `failing_tests` files using regex pattern matching

### 8.2 Limitations

- **Aggregate Coverage**: Coverage data is aggregate, not per-test (limitation of Defects4J)
- **Simplified Matrix**: Coverage matrix assumes all tests cover all lines with hits > 0
- **Class Imbalance**: Still present, though significantly reduced

### 8.3 Future Improvements

- Implement per-test coverage collection using GZoltar
- Align code units across bugs by actual line numbers
- Collect data from more projects for better generalization

---

## Appendix: File Structure

```
combined/
├── combined_coverage_matrix.npy
├── combined_test_labels.npy
├── combined_test_labels.csv
├── combined_bug_info.csv
├── combined_metadata.json
├── visuals/
│   ├── combined_coverage_heatmap_full.png
│   ├── combined_coverage_heatmap_failing_vs_passing.png
│   ├── combined_coverage_distribution.png
│   ├── combined_bug_statistics.png
│   ├── combined_class_balance.png
│   └── combined_coverage_density.png
├── tables/
│   ├── table_summary_statistics.csv
│   ├── table_bug_analysis.csv
│   ├── table_coverage_statistics.csv
│   ├── table_class_imbalance.csv
│   └── table_before_after_comparison.csv
└── COMBINED_REPORT.md (this file)
```

---

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Save report
report_path = os.path.join(OUTPUT_DIR, 'COMBINED_REPORT.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n{'='*60}")
print("✓ Report generated successfully!")
print(f"{'='*60}")
print(f"\nReport saved to: {report_path}")
print(f"\nReport includes:")
print("  - Executive summary")
print("  - Dataset overview")
print("  - Class imbalance analysis")
print("  - Bug-by-bug analysis")
print("  - Visualization references")
print("  - Data files documentation")
print("  - Recommendations")
print("  - Next steps")

