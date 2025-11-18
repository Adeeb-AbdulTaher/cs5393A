"""
DEEPRL4FL Baseline - CNN for Fault Localization
Trains a 1D CNN on coverage matrix to predict failing tests
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("DEEPRL4FL Baseline - CNN Fault Localization Model")
print("=" * 60)

# Load coverage data
print("\n1. Loading data...")
X = np.load('line_coverage_matrix.npy')
y = np.load('line_coverage_labels.npy')

print(f"   Coverage matrix shape: {X.shape}")
print(f"   Labels shape: {y.shape}")
print(f"   Failing tests: {y.sum()}")
print(f"   Passing tests: {len(y) - y.sum()}")

# Reshape for Conv1D: (samples, timesteps, features)
# In our case: (tests, code_lines, 1)
X = X[..., np.newaxis]  # shape (2193, 574, 1)
print(f"   Reshaped for Conv1D: {X.shape}")

# Handle class imbalance: with only 1 failing test, we need special handling
if y.sum() == 1:
    print("\n   ⚠️  Only 1 failing test detected. Using stratified split with special handling...")
    # Use a small test set that includes the failing test
    # Split: 80% train, 20% test, ensuring failing test is in test set
    failing_idx = np.where(y == 1)[0]
    passing_idx = np.where(y == 0)[0]
    
    # Split passing tests: 80% train, 20% test
    passing_train_idx, passing_test_idx = train_test_split(
        passing_idx, test_size=0.2, random_state=42
    )
    
    # Combine: train = passing_train, test = passing_test + failing
    train_idx = np.concatenate([passing_train_idx])
    test_idx = np.concatenate([passing_test_idx, failing_idx])
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"   Train set: {len(X_train)} samples ({y_train.sum()} failing)")
    print(f"   Test set: {len(X_test)} samples ({y_test.sum()} failing)")
else:
    # Normal stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train set: {len(X_train)} samples ({y_train.sum()} failing)")
    print(f"   Test set: {len(X_test)} samples ({y_test.sum()} failing)")

# Define 1D CNN model (DEEPRL4FL style)
print("\n2. Building CNN model...")
model = Sequential([
    Conv1D(16, kernel_size=3, activation='relu', input_shape=(574, 1)),
    Conv1D(32, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classifier
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print("\n3. Model Architecture:")
model.summary()

# Handle class imbalance with class weights
if y_train.sum() > 0:
    total = len(y_train)
    pos = y_train.sum()
    neg = total - pos
    class_weight = {0: total / (2 * neg), 1: total / (2 * pos)}
    print(f"\n4. Class weights: {class_weight}")
else:
    class_weight = None
    print("\n4. No class weights (no failing tests in training set)")

# Train model
print("\n5. Training model...")
print("-" * 60)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight,
    verbose=1
)

# Evaluate on test set
print("\n6. Evaluating on test set...")
print("-" * 60)
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
print(f"   Test Loss: {loss:.4f}")
print(f"   Test Accuracy: {accuracy:.4f}")
print(f"   Test Precision: {precision:.4f}")
print(f"   Test Recall: {recall:.4f}")

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

print("\n7. Classification Report:")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=['Passing', 'Failing']))

print("\n8. Confusion Matrix:")
print("-" * 60)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"   True Negatives: {cm[0,0]}")
print(f"   False Positives: {cm[0,1]}")
print(f"   False Negatives: {cm[1,0]}")
print(f"   True Positives: {cm[1,1]}")

# Save model
print("\n9. Saving model...")
model.save('deeprl4fl_model.h5')
print("   Model saved to 'deeprl4fl_model.h5'")

# Save predictions
np.save('predictions.npy', y_pred)
np.save('test_labels.npy', y_test)
print("   Predictions saved to 'predictions.npy'")

print("\n" + "=" * 60)
print("✓ Training complete!")
print("=" * 60)

