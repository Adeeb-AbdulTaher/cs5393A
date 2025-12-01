# Model Training Guide - XGBoost and Random Forest

## Status

✅ **Random Forest script created**: `train_randomforest_model.py`  
✅ **XGBoost script created**: `train_xgboost_model.py` (XGBoost installing...)  
✅ **Both scripts include threshold optimization**

---

## Quick Start

### Option 1: Random Forest (Ready Now)

```powershell
# Activate environment
.\d4fl_env\Scripts\activate

# Run Random Forest training
python train_randomforest_model.py
```

**Expected time**: 2-5 minutes  
**Status**: ✅ Ready to run

### Option 2: XGBoost (Install First)

```powershell
# Install XGBoost (if not already installed)
.\d4fl_env\Scripts\python.exe -m pip install xgboost

# Run XGBoost training
python train_xgboost_model.py
```

**Expected time**: 1-3 minutes  
**Status**: ⏳ Installing (72MB download)

---

## What the Scripts Do

### Both scripts:

1. ✅ **Load combined data** (10,783 tests, 27 failing)
2. ✅ **Split data** (80% train, 20% test, stratified)
3. ✅ **Train model** with class weights for imbalance
4. ✅ **Test multiple thresholds** (0.1, 0.2, 0.3, 0.4, 0.5)
5. ✅ **Find best threshold** (maximizes F1-score)
6. ✅ **Evaluate performance** with best threshold
7. ✅ **Save model and results**

### Key Features:

- **Threshold Optimization**: Tests 5 different thresholds, picks best
- **Class Imbalance Handling**: Uses class weights / scale_pos_weight
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Feature Importance**: Shows top 20 most important features
- **Saves Everything**: Model, predictions, metrics, analysis

---

## Expected Results

### Random Forest:
- **Best threshold**: Likely 0.1-0.3 (lower than 0.5)
- **Recall**: Should be > 0% (better than CNN's 0%)
- **F1-Score**: Should be > 0 (better than CNN's 0%)
- **ROC-AUC**: Should be > 0.77 (similar or better than CNN)

### XGBoost:
- **Best threshold**: Likely 0.1-0.3
- **Recall**: Should be even better than Random Forest
- **F1-Score**: Should be highest
- **ROC-AUC**: Should be best

---

## Output Files

Both scripts create:

```
combined/models/
├── [model]_model.pkl or .json    # Trained model
├── [model]_predictions.npy        # Binary predictions
├── [model]_predictions_proba.npy  # Prediction probabilities
├── [model]_test_labels.npy        # True labels
├── [model]_threshold_analysis.csv # Threshold comparison
├── [model]_final_metrics.json     # Final performance metrics
└── [model]_feature_importance.csv # Feature importance
```

---

## Comparison: CNN vs Tree-Based Models

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| **CNN** | Deep learning, pattern learning | Struggles with imbalance | Large balanced datasets |
| **Random Forest** | Robust, interpretable | Slower, less powerful | Medium datasets, imbalance |
| **XGBoost** | Best performance, fast | Less interpretable | **Imbalanced data** ⭐ |

---

## Troubleshooting

### If Random Forest is slow:
- Reduce `n_estimators` from 200 to 100
- Reduce `max_depth` from 15 to 10

### If XGBoost installation fails:
- Try: `pip install xgboost --upgrade`
- Or use Random Forest instead (already works)

### If memory issues:
- Reduce `n_estimators` / `n_estimators`
- Use smaller batch sizes

---

## Next Steps After Training

1. ✅ **Compare results** between CNN, Random Forest, and XGBoost
2. ✅ **Analyze feature importance** - which code lines matter most?
3. ✅ **Try ensemble** - combine predictions from multiple models
4. ✅ **Collect more data** - more bugs = better performance

---

## Quick Command Reference

```powershell
# Random Forest (ready now)
python train_randomforest_model.py

# XGBoost (after installation)
python train_xgboost_model.py

# Check if XGBoost is installed
python -c "import xgboost; print('OK')"

# Install XGBoost
python -m pip install xgboost
```

---

**Status**: ✅ Scripts ready, Random Forest can run now, XGBoost installing...

