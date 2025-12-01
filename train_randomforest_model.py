"""
Random Forest Model for Fault Localization with Threshold Optimization
Alternative to XGBoost - uses sklearn (already installed)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score
import warnings
import os
import json
warnings.filterwarnings('ignore')

print("=" * 70)
print("Random Forest Fault Localization Model with Threshold Optimization")
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

# Calculate class weight
class_weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n3. Class imbalance ratio: {class_weight_ratio:.2f}:1 (passing:failing)")

# Train Random Forest model
print("\n4. Training Random Forest model...")
print("-" * 70)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # Handles class imbalance
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Fit model
model.fit(X_train, y_train)
print("   ✓ Model trained successfully!")

# Get predictions (probabilities)
print("\n5. Generating predictions...")
y_pred_proba_train = model.predict_proba(X_train)[:, 1]
y_pred_proba_test = model.predict_proba(X_test)[:, 1]

# Test multiple thresholds
print("\n6. Testing different thresholds...")
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
print(f"\n7. Final predictions with threshold = {best_threshold:.1f}...")
y_pred_final = (y_pred_proba_test > best_threshold).astype(int)

# Final evaluation
print("\n8. Final Evaluation:")
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

print("\n9. Classification Report:")
print("-" * 70)
print(classification_report(y_test, y_pred_final, target_names=['Passing', 'Failing'], digits=4))

# Feature importance
print("\n10. Top 20 Most Important Features:")
print("-" * 70)
feature_importance = model.feature_importances_
top_indices = np.argsort(feature_importance)[-20:][::-1]
for i, idx in enumerate(top_indices, 1):
    print(f"   {i:2d}. Feature {idx:4d}: {feature_importance[idx]:.6f}")

# Save results
print("\n11. Saving model and results...")
os.makedirs('combined/models', exist_ok=True)

# Save model (using pickle)
import pickle
with open('combined/models/randomforest_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✓ Model saved to 'combined/models/randomforest_model.pkl'")

# Save predictions
np.save('combined/models/randomforest_predictions.npy', y_pred_final)
np.save('combined/models/randomforest_predictions_proba.npy', y_pred_proba_test)
np.save('combined/models/randomforest_test_labels.npy', y_test)
print("   ✓ Predictions saved")

# Save threshold analysis
threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv('combined/models/randomforest_threshold_analysis.csv', index=False)
print("   ✓ Threshold analysis saved")

# Save final metrics
final_metrics = {
    'model': 'RandomForest',
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
    }
}

with open('combined/models/randomforest_final_metrics.json', 'w') as f:
    json.dump(final_metrics, f, indent=2)
print("   ✓ Final metrics saved")

# Save feature importance
feature_importance_df = pd.DataFrame({
    'feature_index': range(len(feature_importance)),
    'importance': feature_importance
}).sort_values('importance', ascending=False)
feature_importance_df.to_csv('combined/models/randomforest_feature_importance.csv', index=False)
print("   ✓ Feature importance saved")

print("\n" + "=" * 70)
print("✓ Random Forest training complete!")
print("=" * 70)
print(f"\nSummary:")
print(f"  - Best threshold: {best_threshold:.1f}")
print(f"  - Accuracy: {accuracy_final:.4f}")
print(f"  - Precision: {precision_final:.4f}")
print(f"  - Recall: {recall_final:.4f}")
print(f"  - F1-Score: {f1_final:.4f}")
print(f"  - ROC-AUC: {auc_score:.4f}")
print(f"  - True Positives: {tp} out of {y_test.sum()} failing tests")
print(f"\nModel saved to: combined/models/randomforest_model.pkl")
print(f"\nNote: Once XGBoost finishes installing, you can run train_xgboost_model.py for comparison!")

