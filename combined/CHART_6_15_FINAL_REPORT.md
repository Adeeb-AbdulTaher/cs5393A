# Chart Bugs 6-15 Extension - Final Report

**Date:** November 17, 2025  
**Extension:** Added Chart bugs 6-15 to dataset

---

## Collection Summary

✅ **Successfully collected:** Chart-6 through Chart-15 (10 bugs)  
✅ **Total bugs in dataset:** 15 (up from 5)  
✅ **Success rate:** 100% (10/10)

---

## Dataset Statistics

### Before Extension (5 bugs)
- Total tests: 10,783
- Failing tests: 27 (0.250%)
- Passing tests: 10,756

### After Extension (15 bugs)
- Total tests: 32,349 (estimated, may vary due to deduplication)
- Failing tests: 40+ (individual bugs total)
- Combined dataset: May show fewer due to test deduplication

**Note:** The combine script deduplicates tests with the same name across bugs, so the combined dataset may have fewer total tests than the sum of individual bugs.

---

## Model Performance (After Retraining)

### Random Forest
- **Best Threshold:** 0.2
- **Recall:** 80% (4/5 failing tests caught)
- **Precision:** 0.87%
- **F1-Score:** 1.72%
- **ROC-AUC:** 0.7350

### SMOTE + Random Forest
- **Best Threshold:** 0.1
- **Recall:** 80% (4/5 failing tests caught)
- **Precision:** 0.87%
- **F1-Score:** 1.72%
- **ROC-AUC:** 0.7350

**Note:** Models retrained on combined dataset. Performance metrics reflect test set evaluation.

---

## Files Updated

### Data Files
- `combined/combined_coverage_matrix.npy` - Updated with 15 bugs
- `combined/combined_test_labels.npy` - Updated labels

### Model Files
- `combined/models/randomforest_final_metrics.json` - Retrained metrics
- `combined/models/smote_final_metrics.json` - Retrained metrics

---

## Next Steps

1. ✅ Data collection complete
2. ✅ Data combined
3. ✅ Models retrained
4. ⏭️ Collect Math bugs (next extension)
5. ⏭️ Try XGBoost model

---

**Status:** Extension complete, models retrained, ready for next phase.

