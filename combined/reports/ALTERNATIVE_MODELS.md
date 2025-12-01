# Alternative Models for Fault Localization

## Quick Answer: YES, We Can Use Other Models!

Given the severe class imbalance (398:1), alternative models may perform better than CNN. Here are recommended alternatives:

---

## 1. Tree-Based Models (Highly Recommended)

### XGBoost ⭐ **BEST CHOICE**

**Why**: Excellent for imbalanced data, built-in class weights, fast training

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load data
X = np.load('combined/combined_coverage_matrix.npy')
y = np.load('combined/combined_test_labels.npy')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create DMatrix with scale_pos_weight for imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    eval_metric='auc',
    use_label_encoder=False
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Feature importance
feature_importance = model.feature_importances_
```

**Advantages**:
- ✅ Handles imbalance naturally
- ✅ Fast training
- ✅ Feature importance
- ✅ Good default performance

### LightGBM

Similar to XGBoost but faster:

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    class_weight='balanced'
)

model.fit(X_train, y_train)
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)
```

---

## 2. Deep Learning Alternatives

### Focal Loss (Better for Imbalance)

```python
import tensorflow as tf

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss = -alpha_t * (1 - p_t) ** gamma * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    return focal_loss_fixed

model.compile(
    optimizer='adam',
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy', 'precision', 'recall', 'auc']
)
```

### SMOTE + CNN

```python
from imblearn.over_sampling import SMOTE

# Oversample minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Then train CNN normally
model.fit(X_train_resampled, y_train_resampled, ...)
```

---

## 3. Ensemble Methods

### CNN + XGBoost Ensemble

```python
# Train CNN
cnn_model = ...  # Your CNN
cnn_model.fit(X_train, y_train, ...)

# Extract CNN features (before final layer)
cnn_features = cnn_model.predict(X_train, ...)

# Train XGBoost on CNN features
xgb_model = xgb.XGBClassifier(...)
xgb_model.fit(cnn_features, y_train)

# Predict
cnn_test_features = cnn_model.predict(X_test, ...)
final_predictions = xgb_model.predict(cnn_test_features)
```

### Voting Classifier

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('cnn', cnn_model),
        ('xgb', xgb_model),
        ('rf', rf_model)
    ],
    voting='soft'
)

ensemble.fit(X_train, y_train)
```

---

## 4. Recommended Implementation Order

### Priority 1: XGBoost (Try First!)
- Easiest to implement
- Best performance for imbalanced data
- Fast training

### Priority 2: SMOTE + CNN
- If you want to stick with CNN
- Oversample then train
- Better balance

### Priority 3: Focal Loss
- If you want deep learning
- Better loss function for imbalance
- More complex

### Priority 4: Ensemble
- Combine multiple models
- Best performance
- Most complex

---

## 5. Quick XGBoost Implementation

I can create a script `train_xgboost_model.py` that:
- Loads combined data
- Trains XGBoost with proper class weights
- Evaluates with proper metrics
- Saves model and results

**Would you like me to create this?**

---

## 6. Model Comparison Table

| Model | Pros | Cons | Recommended? |
|-------|------|------|---------------|
| **XGBoost** | Handles imbalance, fast, feature importance | Less interpretable | ✅ **YES** |
| **LightGBM** | Faster than XGBoost | Similar to XGBoost | ✅ Yes |
| **Random Forest** | Interpretable, robust | Slower, less powerful | ⚠️ Maybe |
| **SMOTE + CNN** | Keeps CNN architecture | Extra step, may overfit | ⚠️ Maybe |
| **Focal Loss** | Better loss for imbalance | More complex | ⚠️ Maybe |
| **Ensemble** | Best performance | Complex, slow | ⏭️ Later |

---

## Conclusion

**YES, we can and should try other models!** 

XGBoost is the **best first choice** for our imbalanced data. It's likely to perform much better than CNN for this specific problem.

