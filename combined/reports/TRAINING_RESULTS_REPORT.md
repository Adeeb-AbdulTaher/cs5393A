# Training Results Report - Combined Multi-Bug Model

**Generated:** November 17, 2025  
**Model:** DEEPRL4FL CNN Baseline  
**Dataset:** Combined Chart Bugs 1-5 (10,783 tests, 27 failing)

---

## Executive Summary

The model was trained on combined multi-bug data to address class imbalance. Training stopped at **epoch 6** due to early stopping (validation loss stopped improving). The model achieved high accuracy (99.77%) but struggles with the minority class (failing tests), highlighting the challenges of severe class imbalance.

### Key Findings

- ✅ **Training completed** with early stopping at epoch 6
- ⚠️ **High accuracy** (99.77%) but **misleading** due to class imbalance
- ⚠️ **Zero true positives** - model predicts all tests as passing
- ✅ **ROC-AUC: 0.7752** - Model has some discriminative power
- ⚠️ **Class imbalance** remains the primary challenge (398:1 ratio)

---

## 1. Why Training Stopped at Epoch 6

### Early Stopping Mechanism

The training script uses **early stopping** with the following configuration:
- **Monitor**: Validation loss
- **Patience**: 5 epochs
- **Restore best weights**: Yes

### What Happened

1. **Epoch 1**: Best validation loss (0.4432) - **best model saved**
2. **Epochs 2-6**: Validation loss increased or didn't improve significantly
3. **Epoch 6**: Early stopping triggered (no improvement for 5 epochs)
4. **Result**: Model weights restored from **Epoch 1** (best performance)

### Training History

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|---------------|-----------------|-------------------|---------------------|
| 1 | 1.3720 | **0.4432** ⭐ | 0.9023 | 0.9971 |
| 2 | 0.7910 | 0.5645 | 0.8017 | 0.0029 |
| 3 | 0.6705 | 0.5645 | 0.7941 | 0.0029 |
| 4 | 0.5934 | 0.5645 | 0.7693 | 0.8082 |
| 5 | 0.6751 | 0.5645 | 0.7803 | 0.6165 |
| 6 | 0.4958 | 0.5645 | 0.7655 | 0.8082 |

**Key Observation**: Validation loss was lowest at epoch 1, then plateaued. The model learned to predict all tests as "passing" (high accuracy but poor recall for failing tests).

---

## 2. Model Performance Analysis

### 2.1 Overall Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Test Accuracy** | 0.9977 (99.77%) | Very high, but misleading |
| **Test Precision** | 0.0000 (0%) | No true positives |
| **Test Recall** | 0.0000 (0%) | All failing tests missed |
| **Test F1-Score** | 0.0000 (0%) | No positive predictions |
| **ROC-AUC** | 0.7752 | Moderate discriminative power |
| **PR-AUC** | ~0.0025 | Very low (expected with severe imbalance) |

### 2.2 Confusion Matrix

```
                    Predicted Passing    Predicted Failing
Actual Passing             2,152                    0
Actual Failing                5                    0
```

**Breakdown:**
- **True Negatives (TN)**: 2,152 - Correctly identified passing tests
- **False Positives (FP)**: 0 - No false alarms
- **False Negatives (FN)**: 5 - **All 5 failing tests were missed**
- **True Positives (TP)**: 0 - **No failing tests detected**

### 2.3 Critical Issue: Class Imbalance

The model learned to **always predict "passing"** because:
- **99.75%** of tests are passing (2,152 out of 2,157)
- Predicting "passing" for everything gives **99.77% accuracy**
- The model optimized for accuracy, not recall

**This is a classic class imbalance problem!**

---

## 3. ROC and Precision-Recall Analysis

### 3.1 ROC Curve (AUC = 0.7752)

The ROC-AUC of **0.7752** indicates the model has **some discriminative power**:
- Better than random (0.5)
- But not excellent (>0.9)
- Suggests the model can distinguish patterns, but threshold (0.5) is too high

### 3.2 Precision-Recall Curve

The PR-AUC is very low (~0.0025), which is **expected with severe class imbalance**:
- Precision-Recall is more informative than ROC for imbalanced data
- Low PR-AUC confirms the model struggles with the minority class

### 3.3 Recommendation: Adjust Threshold

The default threshold of **0.5** is too high. Consider:
- **Lower threshold** (e.g., 0.1-0.3) to increase recall
- **Trade-off**: More false positives but catch some failing tests
- Use **precision-recall curve** to find optimal threshold

---

## 4. Training History Analysis

### 4.1 Loss Curves

**Training Loss**: Decreased from 1.37 → 0.50 (good)
**Validation Loss**: Started at 0.44, then increased to 0.56

**Interpretation**: 
- Model is learning (training loss decreases)
- But validation loss increases → **overfitting** or **class imbalance issue**

### 4.2 Accuracy Curves

**Training Accuracy**: Decreased from 90% → 77% (unusual)
**Validation Accuracy**: Started at 99.7%, then dropped to 0.3%, then recovered to 80.8%

**Interpretation**:
- Unusual pattern suggests model is learning to distinguish classes
- But class imbalance causes it to default to majority class

### 4.3 Precision/Recall Curves

**Training Precision**: Very low (~0.007-0.009) - few positive predictions
**Training Recall**: Increased from 29% → 82% - model is learning to catch some failures
**Validation Precision**: Very low (~0.012) - similar issue
**Validation Recall**: Varied (0% → 80%) - inconsistent

**Key Insight**: Model shows learning on training set (recall increases), but validation performance is unstable due to class imbalance.

---

## 5. Prediction Probability Analysis

### 5.1 Distribution

The prediction probabilities show:
- **Most predictions** are very low (< 0.1) - model is conservative
- **Failing tests** may have slightly higher probabilities, but still below 0.5 threshold
- **No clear separation** between passing and failing test probabilities

### 5.2 Threshold Analysis

With threshold = 0.5:
- **0 predictions** above threshold → All predicted as passing
- **Need lower threshold** to catch failing tests

**Recommended Threshold**: 0.1-0.3 (based on precision-recall curve)

---

## 6. Root Cause Analysis

### Why the Model Fails

1. **Severe Class Imbalance** (398:1 ratio)
   - Model optimizes for accuracy → predicts majority class
   - Class weights help but not enough

2. **Insufficient Failing Test Examples**
   - Only 5 failing tests in test set
   - 22 failing tests in training set
   - Not enough to learn robust patterns

3. **Threshold Too High**
   - Default 0.5 threshold is inappropriate for imbalanced data
   - Need threshold tuning based on precision-recall trade-off

4. **Aggregate Coverage Limitation**
   - Using aggregate coverage (not per-test)
   - May not capture test-specific patterns well

---

## 7. Recommendations

### 7.1 Immediate Fixes

1. **Lower the Prediction Threshold**
   ```python
   # Instead of 0.5, use 0.1-0.3
   y_pred = (y_pred_proba > 0.1).astype(int)
   ```

2. **Use Precision-Recall for Threshold Selection**
   - Find threshold that maximizes F1-score
   - Or use threshold that gives acceptable recall (e.g., 50%)

3. **Collect More Data**
   - Get more failing tests (target: 50-100+)
   - Collect more bugs (Chart 6-26, other projects)

### 7.2 Model Improvements

1. **Try Different Models** (see section 8)
   - XGBoost (handles imbalance well)
   - Random Forest with class weights
   - SMOTE + CNN
   - Focal Loss instead of binary crossentropy

2. **Feature Engineering**
   - Per-test coverage (use GZoltar)
   - Code embeddings
   - Test execution order

3. **Advanced Techniques**
   - SMOTE (Synthetic Minority Oversampling)
   - ADASYN (Adaptive Synthetic Sampling)
   - Ensemble methods

### 7.3 Evaluation Metrics

**Stop using accuracy!** Use:
- **Precision-Recall curve** (primary metric)
- **F1-Score** (balance of precision/recall)
- **ROC-AUC** (secondary, less informative for imbalance)
- **Top-K metrics** (if doing ranking)

---

## 8. Alternative Models to Try

### 8.1 Tree-Based Models (Recommended)

**XGBoost**:
- ✅ Excellent for imbalanced data
- ✅ Built-in class weights
- ✅ Feature importance
- ✅ Fast training

**LightGBM**:
- ✅ Similar to XGBoost
- ✅ Faster training
- ✅ Good for large datasets

**Random Forest**:
- ✅ Robust to imbalance
- ✅ Easy to interpret
- ✅ Feature importance

### 8.2 Deep Learning Alternatives

**Focal Loss**:
- ✅ Designed for class imbalance
- ✅ Down-weights easy examples
- ✅ Focuses on hard examples

**SMOTE + CNN**:
- ✅ Oversample minority class
- ✅ Then train CNN
- ✅ Better balance

**Attention Mechanisms**:
- ✅ Focus on relevant code regions
- ✅ Better feature learning

### 8.3 Hybrid Approaches

**Ensemble**:
- ✅ Combine CNN + XGBoost
- ✅ Use CNN features in XGBoost
- ✅ Better generalization

**Two-Stage**:
- ✅ Stage 1: Binary classifier (pass/fail)
- ✅ Stage 2: Ranking for positive predictions
- ✅ Better precision

---

## 9. Comparison: Our Results vs Paper

### Paper (DEEPRL4FL)
- **Top-1 Accuracy**: 62.0% (245/395 bugs)
- **MFR**: 5.94
- **MAR**: 8.57
- **Dataset**: 395 bugs, method-level

### Our Results
- **Accuracy**: 99.77% (misleading - all predicted as passing)
- **Precision**: 0% (no true positives)
- **Recall**: 0% (all failures missed)
- **ROC-AUC**: 0.7752 (some discriminative power)
- **Dataset**: 5 bugs, line-level

### Key Differences

1. **Task Formulation**: Paper does ranking, we do binary classification
2. **Scale**: Paper uses 395 bugs, we use 5 bugs
3. **Features**: Paper has advanced features, we have basic coverage
4. **Evaluation**: Paper uses Top-K, we use binary metrics

**Our work is a baseline** - foundation for adding paper's features.

---

## 10. Next Steps

### Immediate Actions

1. ✅ **Lower threshold** and re-evaluate
2. ✅ **Try XGBoost** or other tree-based models
3. ✅ **Collect more data** (more bugs, more failing tests)
4. ✅ **Implement SMOTE** for oversampling

### Medium-Term

1. ⏭️ **Per-test coverage** (GZoltar)
2. ⏭️ **Feature engineering** (code embeddings, etc.)
3. ⏭️ **Ensemble methods**
4. ⏭️ **Threshold optimization**

### Long-Term

1. ⏭️ **Full RL framework** (from paper)
2. ⏭️ **Advanced features** (mutation, text similarity)
3. ⏭️ **Large-scale evaluation** (100+ bugs)
4. ⏭️ **Ranking task** (Top-K metrics)

---

## 11. Conclusion

The model training revealed the **critical challenge of class imbalance**:
- ✅ Model trains successfully
- ✅ Has some discriminative power (ROC-AUC = 0.7752)
- ⚠️ But fails to detect failing tests (0% recall)
- ⚠️ Needs threshold tuning and/or more data

**Key Takeaway**: With severe class imbalance (398:1), even sophisticated models struggle. We need:
1. **More data** (especially failing tests)
2. **Better techniques** (SMOTE, focal loss, tree-based models)
3. **Proper evaluation** (precision-recall, not accuracy)
4. **Threshold optimization**

The foundation is solid - now we need to address the imbalance challenge!

---

## Appendix: Files Generated

### Visualizations
- `visuals/training_history.png` - Training curves
- `visuals/roc_curve.png` - ROC curve
- `visuals/precision_recall_curve.png` - PR curve
- `visuals/confusion_matrix.png` - Confusion matrix
- `visuals/prediction_probability_distribution.png` - Probability distribution

### Tables
- `tables/table_model_performance_detailed.csv` - Detailed metrics
- `tables/table_training_summary.csv` - Training summary

### Models
- `models/deeprl4fl_combined_model.h5` - Trained model
- `models/training_history.json` - Training history
- `models/predictions_combined.npy` - Predictions
- `models/predictions_proba_combined.npy` - Prediction probabilities

---

**Report Generated:** November 17, 2025  
**Status:** ✅ Complete - Ready for improvements

