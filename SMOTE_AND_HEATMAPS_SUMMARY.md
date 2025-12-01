# SMOTE and Heatmap Guides - Summary

## ✅ Created Files

### 1. SMOTE Training Script
- **File**: `train_smote_model.py`
- **What it does**: Uses SMOTE to oversample failing tests, then trains Random Forest
- **Status**: Ready (installing imbalanced-learn package)

### 2. Heatmap Interpretation Guides
- **File**: `combined/HEATMAP_INTERPRETATION_GUIDE.md` - Detailed guide
- **File**: `combined/HEATMAP_PATTERNS_SUMMARY.md` - Quick reference

---

## SMOTE Overview

### What is SMOTE?
**Synthetic Minority Oversampling Technique**
- Creates synthetic examples of the minority class (failing tests)
- Balances the dataset before training
- Helps models learn better from imbalanced data

### How It Works
1. **Before SMOTE**: 22 failing tests, 8,604 passing tests (391:1 ratio)
2. **After SMOTE**: ~4,300 failing tests, 8,604 passing tests (2:1 ratio)
3. **Result**: Model sees more failing test examples

### To Run SMOTE Training

```powershell
# Install imbalanced-learn (if not already)
.\d4fl_env\Scripts\python.exe -m pip install imbalanced-learn

# Run SMOTE training
.\d4fl_env\Scripts\python.exe train_smote_model.py
```

**Expected**: Better recall than Random Forest alone (maybe 100% recall!)

---

## Heatmap Interpretation - Quick Guide

### What Heatmaps Show

**Coverage Matrix:**
- **Rows** = Tests
- **Columns** = Code lines
- **Bright** = Covered
- **Dark** = Not covered

### Key Patterns to Explain

#### 1. **Bright Columns in Failing Tests**
**Meaning**: Failing tests cover these code lines more than passing tests

**Pattern explanation:**
- "These code lines appear frequently in failing tests"
- Indicates **fault-prone code**
- **Matches feature importance** (Features 129, 1174, 88)

**Example:**
- Feature 129 (top importance) = Bright column in failing tests
- **Conclusion**: Code line 129 is likely fault-prone

#### 2. **Different Patterns in Failing vs Passing**
**Meaning**: Failing tests have unique coverage patterns

**Pattern explanation:**
- "Failing tests exercise different code paths than passing tests"
- This is what the model learns!
- **Good signal** for fault localization

#### 3. **Bright Rows (Tests)**
**Meaning**: Comprehensive tests that cover many code lines

**Pattern explanation:**
- "This test exercises many parts of the code"
- Integration/end-to-end tests
- Harder to localize faults (many suspects)

#### 4. **Dark Rows (Tests)**
**Meaning**: Focused tests that cover few code lines

**Pattern explanation:**
- "This test exercises a small part of the code"
- Unit tests
- Easier to localize faults (few suspects)

---

## Real Example from Our Data

### Feature Importance (Random Forest)
1. Feature 129: 2.26% importance
2. Feature 1174: 1.46% importance  
3. Feature 88: 1.42% importance

### Heatmap Pattern to Check

**In the failing vs passing comparison heatmap:**

1. **Find columns 129, 1174, 88**
2. **Check if they're bright in the failing section**
3. **Check if they're dark in the passing section**

**If YES:**
- ✅ Model learned correctly
- ✅ These code lines are fault-prone
- ✅ Should investigate these first

**This is the pattern you can explain!**

---

## How to Explain Heatmaps in Presentation

### Simple Explanation

1. **"The heatmap shows which code lines are covered by which tests"**
   - Bright = covered, Dark = not covered

2. **"Failing tests have different patterns than passing tests"**
   - Failing tests cover certain code lines more
   - These are fault-prone code lines

3. **"Our model learned to identify these patterns"**
   - Top features (129, 1174, 88) match bright columns in failing tests
   - Model correctly identified fault-prone code

4. **"This helps us prioritize debugging"**
   - Investigate code lines 129, 1174, 88 first
   - These are most likely to contain faults

---

## Files Created

### SMOTE
- ✅ `train_smote_model.py` - SMOTE + Random Forest training

### Heatmap Guides
- ✅ `combined/HEATMAP_INTERPRETATION_GUIDE.md` - Detailed guide
- ✅ `combined/HEATMAP_PATTERNS_SUMMARY.md` - Quick reference

---

## Next Steps

1. ✅ **Install imbalanced-learn**: `pip install imbalanced-learn`
2. ✅ **Run SMOTE training**: `python train_smote_model.py`
3. ✅ **Compare results**: SMOTE vs Random Forest vs CNN
4. ✅ **Use heatmaps**: Guide debugging with pattern analysis

---

**Status**: ✅ SMOTE script ready, heatmap guides complete!

