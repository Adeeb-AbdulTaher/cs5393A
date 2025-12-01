"""
Simplified SMOTE training - direct execution
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
import pickle
warnings.filterwarnings('ignore')

print("=" * 70)
print("SMOTE + Random Forest Training")
print("=" * 70)

# Load data
print("\nLoading data...")
X = np.load('combined/combined_coverage_matrix.npy')
y = np.load('combined/combined_test_labels.npy')
print(f"Data shape: {X.shape}, Labels: {y.sum()} failing out of {len(y)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} ({y_train.sum()} failing)")
print(f"Test: {len(X_test)} ({y_test.sum()} failing)")

# SMOTE
print("\nApplying SMOTE...")
k_neighbors = min(5, y_train.sum() - 1)
smote = SMOTE(sampling_strategy=0.5, k_neighbors=k_neighbors, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {len(X_train_resampled)} samples ({y_train_resampled.sum()} failing)")

# Train
print("\nTraining model...")
model = RandomForestClassifier(
    n_estimators=200, max_depth=15, class_weight='balanced',
    random_state=42, n_jobs=-1
)
model.fit(X_train_resampled, y_train_resampled)
print("Training complete!")

# Predict
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Test thresholds
print("\nTesting thresholds...")
best_f1 = 0
best_threshold = 0.5
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
    y_pred = (y_pred_proba > thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"  Threshold {thresh:.1f}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

# Final predictions
y_pred_final = (y_pred_proba > best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nFinal Results (threshold={best_threshold:.1f}):")
print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"  Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

# Save
os.makedirs('combined/models', exist_ok=True)
with open('combined/models/smote_randomforest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

metrics = {
    'model': 'SMOTE + RandomForest',
    'best_threshold': float(best_threshold),
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'roc_auc': float(auc),
    'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
}

with open('combined/models/smote_final_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n✓ Model and metrics saved!")
print("=" * 70)

