# Heatmap Interpretation Guide

## Understanding Coverage Matrix Heatmaps

This guide explains how to read and interpret the coverage heatmaps generated for fault localization.

---

## 1. What is a Coverage Matrix?

### Structure
- **Rows** = Test cases (each row is one test)
- **Columns** = Code lines/methods (each column is one code unit)
- **Values** = Coverage (1 = covered, 0 = not covered)

### Visual Representation
- **Color intensity** = Coverage level
  - **Bright/Red** = High coverage (many tests cover this line)
  - **Dark/Blue** = Low coverage (few tests cover this line)
  - **White/Empty** = No coverage

---

## 2. Reading the Heatmaps

### 2.1 Full Coverage Heatmap (`combined_coverage_heatmap_full.png`)

**What it shows:**
- Overall coverage pattern across all tests and code lines
- Sample of 500 tests × 500 code lines (subset of full matrix)

**How to read:**
1. **Vertical patterns** (columns):
   - Bright columns = Code lines covered by many tests
   - Dark columns = Code lines rarely covered
   - **Insight**: Bright columns are "popular" code paths

2. **Horizontal patterns** (rows):
   - Bright rows = Tests that cover many code lines
   - Dark rows = Tests that cover few code lines
   - **Insight**: Bright rows are comprehensive tests

3. **Clusters**:
   - Groups of bright cells = Related code covered together
   - **Insight**: Indicates code dependencies

**Patterns to look for:**
- ✅ **Uniform coverage**: Even distribution (good for testing)
- ⚠️ **Sparse coverage**: Many dark areas (potential gaps)
- ✅ **Dense clusters**: Bright regions (well-tested code)

---

### 2.2 Failing vs Passing Comparison (`combined_coverage_heatmap_failing_vs_passing.png`)

**What it shows:**
- Side-by-side comparison of failing tests (top) vs passing tests (bottom)
- 50 failing tests (if available) vs 200 passing tests

**How to read:**
1. **Top section** (Failing tests):
   - Coverage patterns of tests that fail
   - Look for unique patterns

2. **Bottom section** (Passing tests):
   - Coverage patterns of tests that pass
   - Compare with failing tests

3. **Differences**:
   - **Different columns** = Failing tests cover different code lines
   - **Different intensity** = Failing tests have different coverage density
   - **Unique patterns** = Failing tests exercise unique code paths

**Patterns to look for:**
- ✅ **Distinct columns in failing section**: Code lines unique to failing tests
- ✅ **Brighter failing section**: Failing tests cover more code (common pattern)
- ⚠️ **Similar patterns**: Harder to distinguish (model struggles)

---

## 3. Key Patterns and What They Mean

### Pattern 1: Bright Vertical Stripes

**What it means:**
- Certain code lines are covered by many tests
- These are "hot spots" - frequently executed code

**Interpretation:**
- ✅ **Core functionality** - Important code paths
- ⚠️ **Potential fault locations** - If failing tests also cover these

**Example:**
```
Column 129: Very bright (covered by 2000+ tests)
→ This code line is critical and well-tested
→ If failing tests also cover this, it might be fault-prone
```

---

### Pattern 2: Dark Vertical Stripes

**What it means:**
- Certain code lines are rarely covered
- These are "cold spots" - infrequently executed code

**Interpretation:**
- ⚠️ **Untested code** - Potential bugs hidden here
- ⚠️ **Edge cases** - Rare code paths
- ✅ **Dead code** - Unused code (less concerning)

**Example:**
```
Column 500: Very dark (covered by <10 tests)
→ This code line is rarely tested
→ Might contain hidden bugs
```

---

### Pattern 3: Bright Horizontal Stripes

**What it means:**
- Certain tests cover many code lines
- These are "comprehensive" tests

**Interpretation:**
- ✅ **Integration tests** - Test multiple components
- ✅ **End-to-end tests** - Full system coverage
- ⚠️ **Hard to debug** - If they fail, many code lines are suspect

**Example:**
```
Row 100: Very bright (covers 1500+ code lines)
→ This is a comprehensive test
→ If it fails, fault could be in any of 1500 lines
```

---

### Pattern 4: Dark Horizontal Stripes

**What it means:**
- Certain tests cover few code lines
- These are "focused" tests

**Interpretation:**
- ✅ **Unit tests** - Test specific functionality
- ✅ **Easy to debug** - If they fail, fault is in small code region
- ✅ **Good for fault localization** - Narrow down fault location

**Example:**
```
Row 50: Very dark (covers <100 code lines)
→ This is a focused unit test
→ If it fails, fault is likely in those <100 lines
```

---

### Pattern 5: Clusters (Bright Regions)

**What it means:**
- Groups of code lines covered together
- Indicates code dependencies

**Interpretation:**
- ✅ **Related functionality** - Code that works together
- ✅ **Modules/Classes** - Logical code groupings
- ⚠️ **Fault propagation** - Bugs in one area affect related code

**Example:**
```
Columns 100-200: Bright cluster
→ These code lines are always covered together
→ Likely a class or module
→ If one fails, related code might also be affected
```

---

### Pattern 6: Failing Tests Have Different Patterns

**What it means:**
- Failing tests cover different code lines than passing tests
- Unique coverage patterns for failures

**Interpretation:**
- ✅ **Good for fault localization** - Clear signal
- ✅ **Fault-prone code** - Code lines unique to failing tests
- ✅ **Model can learn** - Distinct patterns help ML models

**Example:**
```
Failing tests: Bright in columns 129, 1174, 88
Passing tests: Bright in columns 200, 300, 400
→ Columns 129, 1174, 88 are fault-prone
→ These match our feature importance!
```

---

## 4. Practical Interpretation for Fault Localization

### 4.1 Finding Fault-Prone Code

**Step 1: Identify columns unique to failing tests**
- Look at failing vs passing comparison heatmap
- Find columns that are bright in failing section but dark in passing section
- These are **fault-prone code lines**

**Step 2: Check feature importance**
- Compare with Random Forest feature importance
- Top features should match bright columns in failing tests
- **Confirmation**: If they match, model learned correctly

**Step 3: Prioritize investigation**
- Start with code lines that are:
  - Bright in failing tests
  - High feature importance
  - Rarely covered by passing tests

---

### 4.2 Understanding Test Coverage

**High coverage (bright rows):**
- Comprehensive tests
- Good for finding integration bugs
- Hard to localize faults (many suspects)

**Low coverage (dark rows):**
- Focused tests
- Good for localizing faults (few suspects)
- Might miss integration bugs

**Balance:**
- Mix of both is ideal
- High coverage for integration
- Low coverage for unit testing

---

### 4.3 Code Quality Insights

**Well-tested code (bright columns):**
- Many tests cover this code
- Likely important functionality
- Less likely to have bugs (but not guaranteed)

**Untested code (dark columns):**
- Few tests cover this code
- Potential for hidden bugs
- Should add more tests

**Fault-prone code (bright in failing tests):**
- Code lines that appear in failing tests
- High correlation with failures
- Priority for code review

---

## 5. Real Example from Our Data

### Feature Importance (Random Forest)
1. Feature 129: 2.26% importance
2. Feature 1174: 1.46% importance
3. Feature 88: 1.42% importance

### Heatmap Interpretation
- **If columns 129, 1174, 88 are bright in failing tests:**
  - ✅ Model learned correctly
  - ✅ These are fault-prone code lines
  - ✅ Should investigate these lines first

- **If columns 129, 1174, 88 are dark:**
  - ⚠️ Model might be learning noise
  - ⚠️ Need to verify feature importance
  - ⚠️ Consider feature engineering

---

## 6. Common Patterns Summary

| Pattern | Visual | Meaning | Action |
|---------|--------|---------|--------|
| **Bright column** | Vertical bright stripe | Many tests cover this line | Check if fault-prone |
| **Dark column** | Vertical dark stripe | Few tests cover this line | Add more tests |
| **Bright row** | Horizontal bright stripe | Test covers many lines | Integration test |
| **Dark row** | Horizontal dark stripe | Test covers few lines | Unit test |
| **Bright cluster** | Bright region | Related code | Check dependencies |
| **Failing unique** | Different in failing section | Fault-prone code | **Investigate first!** |

---

## 7. Tips for Analysis

### Do's ✅
- Compare failing vs passing sections
- Look for unique patterns in failing tests
- Match heatmap patterns with feature importance
- Focus on bright columns in failing tests
- Check for clusters (related code)

### Don'ts ❌
- Don't ignore dark areas (might be untested bugs)
- Don't assume bright = good (might be fault-prone)
- Don't look at heatmap in isolation (use with metrics)
- Don't ignore sparse coverage (potential gaps)

---

## 8. Using Heatmaps for Debugging

### Step-by-Step Process

1. **Identify failing test** from test results
2. **Find test row** in heatmap (if available)
3. **Check coverage pattern** - which columns are bright?
4. **Compare with passing tests** - are patterns different?
5. **Check feature importance** - do they match?
6. **Investigate code lines** - start with high importance + bright in failing tests

### Example Workflow

```
1. Test "test2947660" fails
2. Find row in heatmap (if shown)
3. See bright columns: 129, 1174, 88
4. Check: Are these bright in failing tests? Yes!
5. Check feature importance: Top 3 features match!
6. Investigate code lines 129, 1174, 88 first
```

---

## 9. Limitations

### What Heatmaps Don't Show
- ❌ **Execution order** - Which code runs first?
- ❌ **Dependencies** - Which code calls which?
- ❌ **Data flow** - How data moves through code?
- ❌ **Per-test coverage** - Aggregate only (our limitation)

### What Heatmaps Do Show
- ✅ **Coverage patterns** - Which code is covered
- ✅ **Test coverage** - How comprehensive tests are
- ✅ **Fault-prone regions** - Code unique to failures
- ✅ **Coverage gaps** - Untested code

---

## 10. Conclusion

Heatmaps are **visual tools** that help:
- ✅ Understand coverage patterns
- ✅ Identify fault-prone code
- ✅ Validate model learning
- ✅ Guide debugging efforts

**Key Takeaway**: Look for **unique patterns in failing tests** - these indicate fault-prone code that should be investigated first!

---

**Next Steps:**
1. Compare heatmap patterns with feature importance
2. Investigate code lines that are bright in failing tests
3. Add tests for dark columns (untested code)
4. Use heatmaps to guide code review

