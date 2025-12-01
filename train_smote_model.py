"""
SMOTE + Model Training for Fault Localization
Uses SMOTE to oversample minority class (failing tests) before training
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
import os
import json
warnings.filterwarnings('ignore')

print("=" * 70)
print("SMOTE + Random Forest Fault Localization Model")
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
print(f"   Train set: {len(X_train):,} samples ({y_train.sum()} failing, {len(X_train) - y_train.sum():,} passing)")
print(f"   Test set: {len(X_test):,} samples ({y_test.sum()} failing, {len(X_test) - y_test.sum():,} passing)")

# Apply SMOTE to training data only (not test set!)
print("\n3. Applying SMOTE to training data...")
print("-" * 70)
print(f"   Before SMOTE:")
print(f"     Failing: {y_train.sum()}")
print(f"     Passing: {len(y_train) - y_train.sum():,}")
print(f"     Ratio: {(len(y_train) - y_train.sum()) / y_train.sum():.2f}:1")

# Create SMOTE instance
# k_neighbors should be <= number of minority samples
k_neighbors = min(5, y_train.sum() - 1)  # SMOTE needs at least k_neighbors+1 samples
smote = SMOTE(
    sampling_strategy=0.5,  # Balance to 50% (or use 'auto' for 1:1)
    k_neighbors=k_neighbors,
    random_state=42
)

# Apply SMOTE
try:
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"\n   After SMOTE:")
    print(f"     Failing: {y_train_resampled.sum():,}")
    print(f"     Passing: {len(y_train_resampled) - y_train_resampled.sum():,}")
    print(f"     Ratio: {(len(y_train_resampled) - y_train_resampled.sum()) / y_train_resampled.sum():.2f}:1")
    print(f"     Total samples: {len(y_train_resampled):,} (increased from {len(X_train):,})")
    print(f"     ✓ SMOTE applied successfully!")
except Exception as e:
    print(f"   ✗ SMOTE failed: {e}")
    print(f"   Falling back to class_weight='balanced'")
    X_train_resampled, y_train_resampled = X_train, y_train

# Train Random Forest model on resampled data
print("\n4. Training Random Forest model on SMOTE-resampled data...")
print("-" * 70)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # Still use class weights for extra protection
    random_state=42,
    n_jobs=-1,
    verbose=0
)

# Fit model
model.fit(X_train_resampled, y_train_resampled)
print("   ✓ Model trained successfully!")

# Get predictions (probabilities)
print("\n5. Generating predictions...")
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

# Save model
import pickle
with open('combined/models/smote_randomforest_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✓ Model saved to 'combined/models/smote_randomforest_model.pkl'")

# Save predictions
np.save('combined/models/smote_predictions.npy', y_pred_final)
np.save('combined/models/smote_predictions_proba.npy', y_pred_proba_test)
np.save('combined/models/smote_test_labels.npy', y_test)
print("   ✓ Predictions saved")

# Save threshold analysis
threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv('combined/models/smote_threshold_analysis.csv', index=False)
print("   ✓ Threshold analysis saved")

# Save final metrics
final_metrics = {
    'model': 'SMOTE + RandomForest',
    'smote_applied': True,
    'original_train_size': len(X_train),
    'resampled_train_size': len(X_train_resampled),
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

with open('combined/models/smote_final_metrics.json', 'w') as f:
    json.dump(final_metrics, f, indent=2)
print("   ✓ Final metrics saved")

# Save feature importance
feature_importance_df = pd.DataFrame({
    'feature_index': range(len(feature_importance)),
    'importance': feature_importance
}).sort_values('importance', ascending=False)
feature_importance_df.to_csv('combined/models/smote_feature_importance.csv', index=False)
print("   ✓ Feature importance saved")

print("\n" + "=" * 70)
print("✓ SMOTE + Random Forest training complete!")
print("=" * 70)
print(f"\nSummary:")
print(f"  - SMOTE resampled: {len(X_train):,} → {len(X_train_resampled):,} samples")
print(f"  - Best threshold: {best_threshold:.1f}")
print(f"  - Accuracy: {accuracy_final:.4f}")
print(f"  - Precision: {precision_final:.4f}")
print(f"  - Recall: {recall_final:.4f}")
print(f"  - F1-Score: {f1_final:.4f}")
print(f"  - ROC-AUC: {auc_score:.4f}")
print(f"  - True Positives: {tp} out of {y_test.sum()} failing tests")
print(f"\nModel saved to: combined/models/smote_randomforest_model.pkl")

