"""
Analyze training results and generate comprehensive report with graphs and tables
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR
VISUALS_DIR = os.path.join(OUTPUT_DIR, 'visuals')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 70)
print("Analyzing Training Results")
print("=" * 70)

# Load training results
print("\n1. Loading training results...")
try:
    history = json.load(open(os.path.join(MODELS_DIR, 'training_history.json')))
    y_test = np.load(os.path.join(MODELS_DIR, 'test_labels_combined.npy'))
    y_pred = np.load(os.path.join(MODELS_DIR, 'predictions_combined.npy'))
    y_pred_proba = np.load(os.path.join(MODELS_DIR, 'predictions_proba_combined.npy'))
    print("   ✓ Loaded training history and predictions")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# 1. Training History Visualization
print("\n2. Creating training history plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Loss
axes[0, 0].plot(history['loss'], label='Training Loss', marker='o')
axes[0, 0].plot(history['val_loss'], label='Validation Loss', marker='s')
axes[0, 0].set_title('Model Loss Over Epochs', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(history['accuracy'], label='Training Accuracy', marker='o')
axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
axes[0, 1].set_title('Model Accuracy Over Epochs', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history['precision'], label='Training Precision', marker='o')
axes[1, 0].plot(history['val_precision'], label='Validation Precision', marker='s')
axes[1, 0].set_title('Model Precision Over Epochs', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history['recall'], label='Training Recall', marker='o')
axes[1, 1].plot(history['val_recall'], label='Validation Recall', marker='s')
axes[1, 1].set_title('Model Recall Over Epochs', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: visuals/training_history.png")

# 2. ROC Curve
print("\n3. Creating ROC curve...")
from sklearn.metrics import roc_auc_score, roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Fault Localization Model', fontweight='bold', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'roc_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: visuals/roc_curve.png")

# 3. Precision-Recall Curve
print("\n4. Creating Precision-Recall curve...")
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
from sklearn.metrics import auc
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve - Fault Localization Model', fontweight='bold', fontsize=14)
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'precision_recall_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: visuals/precision_recall_curve.png")

# 4. Confusion Matrix Heatmap
print("\n5. Creating confusion matrix heatmap...")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, 
                     index=['Actual Passing', 'Actual Failing'],
                     columns=['Predicted Passing', 'Predicted Failing'])

plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix - Fault Localization Model', fontweight='bold', fontsize=14)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: visuals/confusion_matrix.png")

# 5. Prediction Probability Distribution
print("\n6. Creating prediction probability distribution...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# All predictions
axes[0].hist(y_pred_proba, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
axes[0].set_xlabel('Prediction Probability', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Prediction Probability Distribution (All Tests)', fontweight='bold', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# By class
axes[1].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Passing Tests', color='green')
axes[1].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Failing Tests', color='red')
axes[1].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
axes[1].set_xlabel('Prediction Probability', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Prediction Probability by Class', fontweight='bold', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'prediction_probability_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: visuals/prediction_probability_distribution.png")

# 6. Create detailed metrics table
print("\n7. Creating metrics tables...")
metrics_data = {
    'Metric': [
        'Accuracy',
        'Precision',
        'Recall',
        'F1-Score',
        'ROC-AUC',
        'PR-AUC',
        'True Positives (TP)',
        'True Negatives (TN)',
        'False Positives (FP)',
        'False Negatives (FN)',
        'Total Predictions'
    ],
    'Value': [
        f"{(y_test == y_pred).mean():.4f}",
        f"{cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0:.4f}",
        f"{cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0:.4f}",
        f"{2 * (cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0) * (cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0) / ((cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0) + (cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0)) if ((cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0) + (cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0)) > 0 else 0:.4f}",
        f"{auc_score:.4f}",
        f"{pr_auc:.4f}",
        f"{cm[1,1]}",
        f"{cm[0,0]:,}",
        f"{cm[0,1]}",
        f"{cm[1,0]}",
        f"{len(y_test):,}"
    ]
}
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(os.path.join(TABLES_DIR, 'table_model_performance_detailed.csv'), index=False)
print("   ✓ Saved: tables/table_model_performance_detailed.csv")

# Training summary table
training_summary = {
    'Metric': [
        'Total Epochs Trained',
        'Best Epoch (Early Stopping)',
        'Final Training Loss',
        'Final Validation Loss',
        'Final Training Accuracy',
        'Final Validation Accuracy',
        'Final Training Precision',
        'Final Validation Precision',
        'Final Training Recall',
        'Final Validation Recall',
        'Final Training AUC',
        'Final Validation AUC'
    ],
    'Value': [
        len(history['loss']),
        f"Epoch {np.argmin(history['val_loss']) + 1}",
        f"{history['loss'][-1]:.4f}",
        f"{history['val_loss'][-1]:.4f}",
        f"{history['accuracy'][-1]:.4f}",
        f"{history['val_accuracy'][-1]:.4f}",
        f"{history['precision'][-1]:.4f}",
        f"{history['val_precision'][-1]:.4f}",
        f"{history['recall'][-1]:.4f}",
        f"{history['val_recall'][-1]:.4f}",
        f"{history['auc'][-1]:.4f}",
        f"{history['val_auc'][-1]:.4f}"
    ]
}
training_summary_df = pd.DataFrame(training_summary)
training_summary_df.to_csv(os.path.join(TABLES_DIR, 'table_training_summary.csv'), index=False)
print("   ✓ Saved: tables/table_training_summary.csv")

print("\n" + "=" * 70)
print("✓ Analysis complete!")
print("=" * 70)
print(f"\nGenerated:")
print(f"  - 5 visualizations in visuals/")
print(f"  - 2 tables in tables/")

