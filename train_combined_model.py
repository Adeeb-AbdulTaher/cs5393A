"""
DEEPRL4FL - CNN for Fault Localization on Combined Multi-Bug Data
Trains a 1D CNN on combined coverage matrix to predict failing tests
Uses data from combined/ folder (5 bugs, 10,783 tests, 27 failing)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import warnings
import os
warnings.filterwarnings('ignore')

print("=" * 70)
print("DEEPRL4FL - CNN Fault Localization Model (Combined Multi-Bug Data)")
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

# Reshape for Conv1D: (samples, timesteps, features)
# In our case: (tests, code_lines, 1)
X = X[..., np.newaxis]  # shape (10783, 1808, 1)
print(f"   Reshaped for Conv1D: {X.shape}")

# Stratified train-test split
print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train set: {len(X_train):,} samples ({y_train.sum()} failing, {len(y_train) - y_train.sum():,} passing)")
print(f"   Test set: {len(X_test):,} samples ({y_test.sum()} failing, {len(y_test) - y_test.sum():,} passing)")

# Calculate class weights for imbalanced data
total = len(y_train)
pos = y_train.sum()
neg = total - pos
class_weight = {0: total / (2 * neg), 1: total / (2 * pos)}
print(f"\n3. Class weights: {class_weight}")
print(f"   Weight ratio (failing:passing): {class_weight[1]/class_weight[0]:.2f}:1")

# Define improved 1D CNN model (DEEPRL4FL style)
print("\n4. Building CNN model...")
input_shape = (X.shape[1], 1)  # (1808, 1)

model = Sequential([
    # First convolutional block
    Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
    BatchNormalization(),
    Dropout(0.3),
    
    # Second convolutional block
    Conv1D(64, kernel_size=5, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # Third convolutional block
    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # Flatten and dense layers
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classifier
])

# Compile with additional metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

print("\n5. Model Architecture:")
model.summary()

# Calculate total parameters
total_params = model.count_params()
print(f"\n   Total parameters: {total_params:,}")

# Train model with early stopping and callbacks
print("\n6. Training model...")
print("-" * 70)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate on test set
print("\n7. Evaluating on test set...")
print("-" * 70)
test_results = model.evaluate(X_test, y_test, verbose=0)
print(f"   Test Loss: {test_results[0]:.4f}")
print(f"   Test Accuracy: {test_results[1]:.4f}")
print(f"   Test Precision: {test_results[2]:.4f}")
print(f"   Test Recall: {test_results[3]:.4f}")
print(f"   Test AUC: {test_results[4]:.4f}")

# Predictions
print("\n8. Generating predictions...")
y_pred_proba = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate additional metrics
try:
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"   ROC-AUC Score: {auc_score:.4f}")
except Exception as e:
    print(f"   ROC-AUC calculation skipped: {e}")

print("\n9. Classification Report:")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=['Passing', 'Failing'], digits=4))

print("\n10. Confusion Matrix:")
print("-" * 70)
cm = confusion_matrix(y_test, y_pred)
print(f"   Confusion Matrix:")
print(f"   {'':>15} {'Predicted Passing':>20} {'Predicted Failing':>20}")
print(f"   {'Actual Passing':>15} {cm[0,0]:>20} {cm[0,1]:>20}")
print(f"   {'Actual Failing':>15} {cm[1,0]:>20} {cm[1,1]:>20}")
print(f"\n   True Negatives (TN): {cm[0,0]:,}")
print(f"   False Positives (FP): {cm[0,1]:,}")
print(f"   False Negatives (FN): {cm[1,0]:,}")
print(f"   True Positives (TP): {cm[1,1]:,}")

# Calculate additional metrics
if cm[1,1] + cm[0,1] > 0:
    precision = cm[1,1] / (cm[1,1] + cm[0,1])
    print(f"   Precision: {precision:.4f}")
if cm[1,1] + cm[1,0] > 0:
    recall = cm[1,1] / (cm[1,1] + cm[1,0])
    print(f"   Recall: {recall:.4f}")
if cm[1,1] + cm[0,1] > 0 and cm[1,1] + cm[1,0] > 0:
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"   F1-Score: {f1:.4f}")

# Save model
print("\n11. Saving model and results...")
os.makedirs('combined/models', exist_ok=True)
model.save('combined/models/deeprl4fl_combined_model.h5')
print("   Model saved to 'combined/models/deeprl4fl_combined_model.h5'")

# Save predictions and history
np.save('combined/models/predictions_combined.npy', y_pred)
np.save('combined/models/predictions_proba_combined.npy', y_pred_proba)
np.save('combined/models/test_labels_combined.npy', y_test)

# Save training history
import json
history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
with open('combined/models/training_history.json', 'w') as f:
    json.dump(history_dict, f, indent=2)

print("   Predictions saved")
print("   Training history saved to 'combined/models/training_history.json'")

# Summary
print("\n" + "=" * 70)
print("✓ Training complete!")
print("=" * 70)
print(f"\nSummary:")
print(f"  - Dataset: {X.shape[0]:,} tests ({y.sum()} failing, {len(y) - y.sum():,} passing)")
print(f"  - Train: {len(X_train):,} tests ({y_train.sum()} failing)")
print(f"  - Test: {len(X_test):,} tests ({y_test.sum()} failing)")
print(f"  - Model: {total_params:,} parameters")
print(f"  - Test Accuracy: {test_results[1]:.4f}")
print(f"  - Test Precision: {test_results[2]:.4f}")
print(f"  - Test Recall: {test_results[3]:.4f}")
print(f"  - Test AUC: {test_results[4]:.4f}")
print(f"\nModel saved to: combined/models/deeprl4fl_combined_model.h5")

