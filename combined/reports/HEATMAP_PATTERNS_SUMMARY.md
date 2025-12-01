# Heatmap Patterns - Quick Reference

## Quick Guide to Reading Heatmaps

### What You're Looking At

**Coverage Matrix Heatmap:**
- **Rows** = Tests (each test is a row)
- **Columns** = Code lines (each code line is a column)
- **Color** = Coverage (bright = covered, dark = not covered)

---

## Key Patterns to Explain

### 1. **Bright Vertical Stripes** (Columns)
**What it means:** Many tests cover this code line

**Pattern explanation:**
- "This code line (column) is executed by many different tests"
- Indicates **popular code path** - frequently used functionality
- **If in failing tests**: This code might be fault-prone

**Example from our data:**
- Feature 129 (top importance) = Bright column in failing tests
- Suggests: Code line 129 is covered by failing tests more than passing tests
- **Conclusion**: Line 129 is likely fault-prone

---

### 2. **Dark Vertical Stripes** (Columns)
**What it means:** Few tests cover this code line

**Pattern explanation:**
- "This code line is rarely tested"
- Indicates **untested or edge case code**
- **Risk**: Hidden bugs might be here

**Action:** Add more tests for dark columns

---

### 3. **Bright Horizontal Stripes** (Rows)
**What it means:** This test covers many code lines

**Pattern explanation:**
- "This test is comprehensive - it exercises many parts of the code"
- Indicates **integration or end-to-end test**
- **If it fails**: Fault could be in any of the covered lines

**Example:**
- A test that covers 1500+ lines = Integration test
- Harder to localize fault (many suspects)

---

### 4. **Dark Horizontal Stripes** (Rows)
**What it means:** This test covers few code lines

**Pattern explanation:**
- "This test is focused - it exercises a small part of the code"
- Indicates **unit test**
- **If it fails**: Fault is likely in those few lines

**Example:**
- A test that covers <100 lines = Unit test
- Easier to localize fault (few suspects)

---

### 5. **Failing vs Passing Comparison**

**Key Pattern to Look For:**

**Failing tests have BRIGHT columns that passing tests DON'T have**

**What this means:**
- Failing tests cover code lines that passing tests don't
- These code lines are **fault-prone**
- **This is what the model learns!**

**Example from our results:**
- Feature 129, 1174, 88 are top important features
- **If these columns are bright in failing tests but dark in passing tests:**
  - ✅ Model learned correctly
  - ✅ These are fault-prone code lines
  - ✅ Should investigate these first

---

## Real Example from Our Data

### Feature Importance (Random Forest)
1. **Feature 129**: 2.26% importance ⭐
2. **Feature 1174**: 1.46% importance
3. **Feature 88**: 1.42% importance

### What to Check in Heatmap

1. **Look at failing vs passing comparison heatmap**
2. **Find columns 129, 1174, 88**
3. **Check if they're bright in failing section**
4. **Check if they're dark in passing section**

**If YES:**
- ✅ Model learned correctly
- ✅ These code lines are fault-prone
- ✅ These should be investigated first

**If NO:**
- ⚠️ Model might be learning noise
- ⚠️ Need to verify
- ⚠️ Consider more data or features

---

## Patterns That Indicate Fault-Prone Code

### Pattern 1: Unique to Failing Tests
- **Bright in failing tests**
- **Dark in passing tests**
- **High feature importance**
- **→ FAULT-PRONE CODE** ⚠️

### Pattern 2: High Coverage but Still Fails
- **Bright in both failing and passing**
- **But failing tests have different pattern**
- **→ POTENTIAL FAULT** ⚠️

### Pattern 3: Rarely Covered
- **Dark in both**
- **But appears in failing tests**
- **→ UNTESTED BUG** ⚠️

---

## How to Use This for Debugging

### Step 1: Identify Failing Test
- From test results, get failing test name

### Step 2: Find in Heatmap
- Locate the test's row in heatmap
- See which columns are bright

### Step 3: Compare with Feature Importance
- Check if bright columns match top features
- If yes → Model learned correctly

### Step 4: Investigate
- Start with code lines that are:
  - Bright in failing tests
  - High feature importance
  - Different from passing tests

---

## Summary Table

| Pattern | Visual | Meaning | Action |
|---------|--------|---------|--------|
| Bright column in failing | Vertical bright in failing section | Fault-prone code | **Investigate first!** |
| Dark column | Vertical dark stripe | Untested code | Add tests |
| Bright row | Horizontal bright stripe | Comprehensive test | Integration test |
| Dark row | Horizontal dark stripe | Focused test | Unit test |
| Cluster | Bright region | Related code | Check dependencies |
| Unique to failing | Different patterns | Fault-prone | **Priority investigation** |

---

## Key Takeaway

**The most important pattern:**
- **Code lines that are BRIGHT in failing tests but DARK in passing tests**
- These are **fault-prone code lines**
- These should be **investigated first**
- These should **match feature importance** (if model learned correctly)

---

**See `HEATMAP_INTERPRETATION_GUIDE.md` for detailed explanation!**

