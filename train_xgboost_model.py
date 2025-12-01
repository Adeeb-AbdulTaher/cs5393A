"""
XGBoost Model for Fault Localization with Threshold Optimization
Uses combined multi-bug data and tests multiple thresholds
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score
import warnings
import os
import json
warnings.filterwarnings('ignore')

print("=" * 70)
print("XGBoost Fault Localization Model with Threshold Optimization")
print("=" * 70)

# Load combined coverage data
print("\n1. Loading combined data...")
X = np.load('combined/combined_coverage_matrix.npy')
y = np.load('combined/combined_test_labels.npy')

print(f"   Coverage matrix shape: {X.shape}")
print(f"   Labels shape: {y.shape}")
print(f"   Total tests: {len(y):,}")
print(f"   Failing tests: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
print(f"   Passing tests: {len(y) - y.sum():,} ({(len(y) - y.sum())/len(y)*100:.2f}%)")

# Stratified train-test split
print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train set: {len(X_train):,} samples ({y_train.sum()} failing)")
print(f"   Test set: {len(X_test):,} samples ({y_test.sum()} failing)")

# Calculate scale_pos_weight for XGBoost (handles class imbalance)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n3. Class imbalance ratio: {scale_pos_weight:.2f}:1 (passing:failing)")
print(f"   XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

# Train XGBoost model
print("\n4. Training XGBoost model...")
print("-" * 70)

model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

# Fit model
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

print("\n5. Model trained successfully!")
print(f"   Best iteration: {model.best_iteration}")
print(f"   Best score: {model.best_score:.4f}")

# Get predictions (probabilities)
print("\n6. Generating predictions...")
y_pred_proba_train = model.predict_proba(X_train)[:, 1]
y_pred_proba_test = model.predict_proba(X_test)[:, 1]

# Test multiple thresholds
print("\n7. Testing different thresholds...")
print("-" * 70)

thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5]
threshold_results = []

for threshold in thresholds_to_test:
    y_pred = (y_pred_proba_test > threshold).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    threshold_results.append({
        'threshold': threshold,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy
    })
    
    print(f"\n   Threshold = {threshold:.1f}:")
    print(f"     TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"     Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

# Find best threshold (maximize F1-score)
best_threshold_result = max(threshold_results, key=lambda x: x['f1_score'])
best_threshold = best_threshold_result['threshold']

print(f"\n   ⭐ Best threshold: {best_threshold:.1f} (F1-score: {best_threshold_result['f1_score']:.4f})")

# Use best threshold for final predictions
print(f"\n8. Final predictions with threshold = {best_threshold:.1f}...")
y_pred_final = (y_pred_proba_test > best_threshold).astype(int)

# Final evaluation
print("\n9. Final Evaluation:")
print("-" * 70)

# Calculate metrics
cm_final = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm_final.ravel()

precision_final = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_final = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_final = 2 * (precision_final * recall_final) / (precision_final + recall_final) if (precision_final + recall_final) > 0 else 0
accuracy_final = (tp + tn) / (tp + tn + fp + fn)
auc_score = roc_auc_score(y_test, y_pred_proba_test)

print(f"   Confusion Matrix:")
print(f"   {'':>15} {'Predicted Passing':>20} {'Predicted Failing':>20}")
print(f"   {'Actual Passing':>15} {tn:>20} {fp:>20}")
print(f"   {'Actual Failing':>15} {fn:>20} {tp:>20}")
print(f"\n   True Positives (TP): {tp}")
print(f"   True Negatives (TN): {tn:,}")
print(f"   False Positives (FP): {fp}")
print(f"   False Negatives (FN): {fn}")
print(f"\n   Accuracy: {accuracy_final:.4f}")
print(f"   Precision: {precision_final:.4f}")
print(f"   Recall: {recall_final:.4f}")
print(f"   F1-Score: {f1_final:.4f}")
print(f"   ROC-AUC: {auc_score:.4f}")

print("\n10. Classification Report:")
print("-" * 70)
print(classification_report(y_test, y_pred_final, target_names=['Passing', 'Failing'], digits=4))

# Feature importance
print("\n11. Top 20 Most Important Features:")
print("-" * 70)
feature_importance = model.feature_importances_
top_indices = np.argsort(feature_importance)[-20:][::-1]
for i, idx in enumerate(top_indices, 1):
    print(f"   {i:2d}. Feature {idx:4d}: {feature_importance[idx]:.6f}")

# Save results
print("\n12. Saving model and results...")
os.makedirs('combined/models', exist_ok=True)

# Save model
model.save_model('combined/models/xgboost_model.json')
print("   ✓ Model saved to 'combined/models/xgboost_model.json'")

# Save predictions
np.save('combined/models/xgboost_predictions.npy', y_pred_final)
np.save('combined/models/xgboost_predictions_proba.npy', y_pred_proba_test)
np.save('combined/models/xgboost_test_labels.npy', y_test)
print("   ✓ Predictions saved")

# Save threshold analysis
threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv('combined/models/xgboost_threshold_analysis.csv', index=False)
print("   ✓ Threshold analysis saved to 'combined/models/xgboost_threshold_analysis.csv'")

# Save final metrics
final_metrics = {
    'best_threshold': float(best_threshold),
    'accuracy': float(accuracy_final),
    'precision': float(precision_final),
    'recall': float(recall_final),
    'f1_score': float(f1_final),
    'roc_auc': float(auc_score),
    'confusion_matrix': {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    },
    'scale_pos_weight': float(scale_pos_weight),
    'best_iteration': int(model.best_iteration),
    'best_score': float(model.best_score)
}

with open('combined/models/xgboost_final_metrics.json', 'w') as f:
    json.dump(final_metrics, f, indent=2)
print("   ✓ Final metrics saved to 'combined/models/xgboost_final_metrics.json'")

# Save feature importance
feature_importance_df = pd.DataFrame({
    'feature_index': range(len(feature_importance)),
    'importance': feature_importance
}).sort_values('importance', ascending=False)
feature_importance_df.to_csv('combined/models/xgboost_feature_importance.csv', index=False)
print("   ✓ Feature importance saved to 'combined/models/xgboost_feature_importance.csv'")

print("\n" + "=" * 70)
print("✓ XGBoost training complete!")
print("=" * 70)
print(f"\nSummary:")
print(f"  - Best threshold: {best_threshold:.1f}")
print(f"  - Accuracy: {accuracy_final:.4f}")
print(f"  - Precision: {precision_final:.4f}")
print(f"  - Recall: {recall_final:.4f}")
print(f"  - F1-Score: {f1_final:.4f}")
print(f"  - ROC-AUC: {auc_score:.4f}")
print(f"  - True Positives: {tp} out of {y_test.sum()} failing tests")
print(f"\nModel saved to: combined/models/xgboost_model.json")

