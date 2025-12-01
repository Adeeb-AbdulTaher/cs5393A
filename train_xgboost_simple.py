"""Simplified XGBoost training"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
import os
import json

print("=" * 70)
print("XGBoost Training")
print("=" * 70)

# Load data
print("\nLoading data...")
X = np.load('combined/combined_coverage_matrix.npy')
y = np.load('combined/combined_test_labels.npy')
print(f"Data: {X.shape}, Failing: {y.sum()}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} ({y_train.sum()} failing)")
print(f"Test: {len(X_test):,} ({y_test.sum()} failing)")

# Train
print("\nTraining XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class weight: {scale_pos_weight:.2f}")

model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    eval_metric='auc',
    use_label_encoder=False
)

model.fit(X_train, y_train, verbose=False)
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
    print(f"  Threshold {thresh:.1f}: Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

# Final
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
model.save_model('combined/models/xgboost_model.json')

metrics = {
    'model': 'XGBoost',
    'best_threshold': float(best_threshold),
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'roc_auc': float(auc),
    'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
}

with open('combined/models/xgboost_final_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n✓ Model and metrics saved!")

