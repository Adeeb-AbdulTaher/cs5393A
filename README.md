# Defects4J Fault Localization Project

Machine learning-based fault localization using Defects4J dataset with combined multi-bug analysis to address class imbalance.

## Overview

This project implements a deep learning approach for fault localization using code coverage data from the Defects4J benchmark. The main contribution is addressing severe class imbalance by combining data from multiple bugs, achieving a **27x improvement** in failing test count.

## Key Features

- ✅ **Multi-bug data collection** from Defects4J Chart project (bugs 1-5)
- ✅ **Coverage matrix generation** for machine learning input
- ✅ **Class imbalance mitigation** (1 → 27 failing tests)
- ✅ **Comprehensive analysis** with tables and visualizations
- ✅ **Deep learning model** (1D CNN) for fault localization
- ✅ **Detailed reporting** with insights and recommendations

## Dataset Statistics

- **Total Bugs**: 5 (Chart-1 through Chart-5)
- **Total Tests**: 10,783
- **Failing Tests**: 27 (0.25%)
- **Passing Tests**: 10,756 (99.75%)
- **Code Units**: 1,808 lines
- **Coverage Matrix**: 10,783 × 1,808

## Project Structure

```
project/
├── combined/                    # Combined multi-bug analysis
│   ├── combine_multi_bug_data.py
│   ├── generate_graphs.py
│   ├── generate_tables.py
│   ├── generate_report.py
│   ├── COMBINED_REPORT.md      # Comprehensive report
│   ├── visuals/                 # Generated visualizations
│   └── tables/                  # Analysis tables
├── create_coverage_matrix.py   # Coverage matrix generation
├── train_deeprl4fl.py          # ML model training
├── visualize_coverage.py       # Visualization tools
└── multi_bug_data/             # Collected bug data
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv d4fl_env
source d4fl_env/bin/activate  # Linux/Mac
# or
d4fl_env\Scripts\activate    # Windows

# Install dependencies
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
```

### 2. Collect Data (WSL)

```bash
# Collect Chart bugs 1-5
bash quick_collect_bugs.sh
```

### 3. Combine Data

```python
# Combine multiple bugs
python combined/combine_multi_bug_data.py
```

### 4. Generate Analysis

```python
# Generate graphs
python combined/generate_graphs.py

# Generate tables
python combined/generate_tables.py

# Generate report
python combined/generate_report.py
```

### 5. Train Model

```python
# Train deep learning model
python train_deeprl4fl.py
```

## Results

### Class Balance Improvement

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Failing Tests | 1 | 27 | **27x** |
| Total Tests | 2,193 | 10,783 | **4.9x** |
| Failing % | 0.046% | 0.25% | **5.4x** |

### Key Findings

1. **Chart_4 is the outlier**: Contains 22 failing tests (81% of total)
2. **Failing tests have higher coverage**: 2.1x more lines covered than passing tests
3. **Class imbalance remains**: 398:1 ratio (passing:failing) - requires class weighting in ML

## Documentation

- **[COMBINED_REPORT.md](combined/COMBINED_REPORT.md)** - Comprehensive analysis report
- **[COLLECT_MORE_DATA_GUIDE.md](COLLECT_MORE_DATA_GUIDE.md)** - Guide for collecting more bugs
- **[DEFECTS4J_GUIDE.md](DEFECTS4J_GUIDE.md)** - Defects4J setup and usage
- **[TRAINING_RESULTS_SUMMARY.md](TRAINING_RESULTS_SUMMARY.md)** - ML model results

## Requirements

- Python 3.8+
- Defects4J framework
- WSL (for Defects4J on Windows)
- Java 11
- Perl with required modules

## Technologies Used

- **Python**: Data processing and ML
- **TensorFlow/Keras**: Deep learning
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Defects4J**: Bug dataset and framework

## Repository

GitHub: [https://github.com/Adeeb-AbdulTaher/cs5393A.git](https://github.com/Adeeb-AbdulTaher/cs5393A.git)

## License

This project is for academic/research purposes.

## Author

Adeeb AbdulTaher

---

**Status**: ✅ Complete - Ready for ML training and evaluation

