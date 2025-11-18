# Tables Analysis - What We Can Create

## Summary

Based on our current setup (Chart-1 bug, DEEPRL4FL baseline), here's what tables we can and cannot create:

## ❌ Cannot Create (Need More Data/Methods)

### TABLE III: Comparative Study
**Why not**: Requires implementing multiple fault localization methods:
- MULTRIC
- FLUCCS  
- TraPT
- DeepFL
- DEEPRL4FL (we have this)

**What we have**: Only DEEPRL4FL baseline

### TABLE IV: Ordering & StateDep Variants
**Why not**: Requires implementing:
- Base version (we have this)
- Base + Order variant
- Base + Order + StateDep variant

**What we have**: Only base version

### TABLE V: Learning Representations
**Why not**: Requires:
- Base (SpecMatrix) - we have this
- Base + New MutMatrix
- Base + New MutMatrix + CodeRep
- Base + New MutMatrix + CodeRep + TextSim

**What we have**: Only base (coverage matrix)

### TABLE VII: ManyBugs vs Defects4J
**Why not**: Requires:
- ManyBugs dataset (C projects)
- Multiple Defects4J bugs (we have 1, need 395+)

**What we have**: 1 Defects4J bug only

## ⚠️ Partial - Can Create Simplified Version

### TABLE VI: Cross-project vs Within-project
**Can create**: Simplified version for Chart project only

**Limitations**:
- Paper shows results for all Chart bugs (26 bugs)
- We only have Chart-1 (1 bug)
- Need to checkout all Chart bugs (1-26) to create full table

**What we can do**:
- Create a single-row table for Chart-1
- Show cross-project vs within-project if we train on other projects and test on Chart

## ✅ Can Create (Simplified Versions)

### 1. Basic Results Table
**File**: `table_basic_results.csv`

Shows:
- Total tests
- Failing/passing test counts
- Coverage statistics
- Average coverage metrics

### 2. Model Performance Table
**File**: `table_model_performance.csv`

Shows:
- Accuracy
- Precision
- Recall
- F1-Score

### 3. Coverage Analysis Table
**File**: `table_coverage_analysis.csv`

Shows:
- Coverage by test type (failing vs passing)
- Average lines covered
- Coverage percentages

### 4. Simplified Results Table
**File**: `table_simplified_results.csv`

Shows:
- Project: Chart
- Bug: Chart-1
- Basic metrics
- Model accuracy

## How to Create Full Tables

### For TABLE VI (Cross-project vs Within-project):

1. **Checkout all Chart bugs**:
```bash
# In WSL
for i in {1..26}; do
  sudo /root/defects4j/framework/bin/defects4j checkout -p Chart -v ${i}b -w /tmp/Chart_${i}b
done
```

2. **Extract coverage for each bug**

3. **Train model**:
   - Cross-project: Train on other projects (Time, Math, etc.), test on Chart
   - Within-project: Train on some Chart bugs, test on others

4. **Calculate metrics**:
   - Top-1, Top-3, Top-5 accuracy
   - MFR (Mean First Rank)
   - MAR (Mean Average Rank)

### For Other Tables:

- **TABLE III**: Implement other FL methods or use published results
- **TABLE IV**: Implement Ordering and StateDep variants
- **TABLE V**: Implement mutation testing, code representation, text similarity
- **TABLE VII**: Get ManyBugs dataset and process multiple Defects4J bugs

## Current Capabilities

With our current setup, we can demonstrate:
- ✅ Complete ML pipeline
- ✅ Coverage matrix creation
- ✅ Model training
- ✅ Basic performance metrics
- ✅ Visualization capabilities

For full paper tables, we would need:
- Multiple bugs (ideally 50+)
- Multiple projects
- Additional features (mutations, code representations)
- Other FL methods for comparison

