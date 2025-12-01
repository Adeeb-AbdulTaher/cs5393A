"""
DEEPRL4FL - CNN for Fault Localization on Combined Multi-Bug Data
25 Epochs Version - Extended Training
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
import os
warnings.filterwarnings('ignore')

print("=" * 70)
print("DEEPRL4FL - CNN Fault Localization (25 Epochs - Extended Training)")
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

# Reshape for Conv1D
X = X[..., np.newaxis]
print(f"   Reshaped for Conv1D: {X.shape}")

# Stratified train-test split
print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train set: {len(X_train):,} samples ({y_train.sum()} failing)")
print(f"   Test set: {len(X_test):,} samples ({y_test.sum()} failing)")

# Calculate class weights
total = len(y_train)
pos = y_train.sum()
neg = total - pos
class_weight = {0: total / (2 * neg), 1: total / (2 * pos)}
print(f"\n3. Class weights: {class_weight}")

# Define 1D CNN model
print("\n4. Building CNN model...")
input_shape = (X.shape[1], 1)

model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
    BatchNormalization(),
    Dropout(0.3),
    Conv1D(64, kernel_size=5, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.AUC(name='auc')]
)

print("\n5. Model Architecture:")
model.summary()

# Callbacks for 25 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

# Train model - 25 epochs
print("\n6. Training model (25 epochs with early stopping)...")
print("-" * 70)

history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate
print("\n7. Evaluating on test set...")
print("-" * 70)
test_results = model.evaluate(X_test, y_test, verbose=0)
print(f"   Test Loss: {test_results[0]:.4f}")
print(f"   Test Accuracy: {test_results[1]:.4f}")
print(f"   Test Precision: {test_results[2]:.4f}")
print(f"   Test Recall: {test_results[3]:.4f}")
print(f"   Test AUC: {test_results[4]:.4f}")

# Predictions
y_pred_proba = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

print("\n8. Classification Report:")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=['Passing', 'Failing'], digits=4))

print("\n9. Confusion Matrix:")
print("-" * 70)
cm = confusion_matrix(y_test, y_pred)
print(f"   TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
print(f"   FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

# Save model
print("\n10. Saving model...")
os.makedirs('combined/models', exist_ok=True)
model.save('combined/models/deeprl4fl_combined_25epochs.h5')
np.save('combined/models/predictions_25epochs.npy', y_pred)
np.save('combined/models/predictions_proba_25epochs.npy', y_pred_proba)
np.save('combined/models/test_labels_25epochs.npy', y_test)

import json
history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
with open('combined/models/training_history_25epochs.json', 'w') as f:
    json.dump(history_dict, f, indent=2)

print("   Model saved to 'combined/models/deeprl4fl_combined_25epochs.h5'")
print("\n" + "=" * 70)
print("✓ Training complete (25 epochs)!")
print("=" * 70)

