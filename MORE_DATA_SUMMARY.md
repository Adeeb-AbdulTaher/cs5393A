# Getting More Data to Address Class Imbalance

## ✅ What I've Created

I've created **3 scripts** and **1 guide** to help you collect more data:

### 1. **`quick_collect_bugs.sh`** ⚡ (Easiest to Start)
- **What it does**: Collects Chart bugs 1-5 automatically
- **How to run**: 
  ```bash
  # In WSL terminal
  bash quick_collect_bugs.sh
  ```
- **Time**: ~20-30 minutes
- **Result**: ~5-10 failing tests (instead of 1)

### 2. **`collect_multiple_bugs.py`** 🐍 (Python Automation)
- **What it does**: Automated Python script to collect bugs from multiple projects
- **How to run**: 
  ```powershell
  python collect_multiple_bugs.py
  ```
- **Configurable**: Edit the script to choose which bugs/projects
- **Time**: Depends on how many bugs you select

### 3. **`combine_multi_bug_data.py`** 🔗 (Combine Data)
- **What it does**: Combines all collected bug data into one dataset
- **How to run**: 
  ```powershell
  python combine_multi_bug_data.py
  ```
- **Output**: 
  - `combined_coverage_matrix.npy` - All tests from all bugs
  - `combined_test_labels.npy` - All labels (failing/passing)
  - `combined_test_labels.csv` - Human-readable format
  - `combined_bug_info.csv` - Summary of collected bugs

### 4. **`COLLECT_MORE_DATA_GUIDE.md`** 📖 (Full Guide)
- Complete instructions and strategies
- Lists all available Defects4J projects and bug counts
- Time estimates and recommendations

## 🎯 Quick Start (Recommended)

### Step 1: Collect 5 Chart Bugs
```bash
# In WSL
bash quick_collect_bugs.sh
```

This will:
- Checkout Chart bugs 1-5
- Run tests for each
- Collect coverage data
- Save to `multi_bug_data/Chart_1/`, `Chart_2/`, etc.

### Step 2: Combine the Data
```powershell
# In PowerShell (Windows)
python combine_multi_bug_data.py
```

### Step 3: Retrain Model
Update `train_deeprl4fl.py` to use combined data:
```python
# Load combined data instead of single bug
X = np.load('combined_coverage_matrix.npy')
y = np.load('combined_test_labels.npy')
```

## 📊 Expected Results

### With 5 Chart Bugs:
- **Before**: 1 failing / 2,193 total (0.046%)
- **After**: ~5-10 failing / ~10,000+ total (~0.1-0.2%)
- **Improvement**: 5-10x more failing tests!

### With 10 Chart Bugs:
- **After**: ~10-20 failing / ~20,000+ total (~0.1-0.2%)

### With Multiple Projects (Chart + Math + Time):
- **After**: 50-100+ failing / 50,000+ total (~0.2-0.5%)

## 📁 Available Data Sources

### Defects4J Projects (854+ bugs total):

| Project | Bugs | Recommended? |
|---------|------|--------------|
| Chart | 26 | ✅ Yes - Start here |
| Math | 106 | ✅ Yes - Large dataset |
| Time | 26 | ✅ Yes - Good size |
| Closure | 174 | ⚠️ Very large |
| Lang | 61 | ✅ Yes - Good size |
| Mockito | 38 | ✅ Yes - Medium |
| **Total** | **854+** | **Lots of data!** |

## ⏱️ Time Estimates

- **Per bug**: 2-5 minutes
- **5 bugs**: 20-30 minutes
- **10 bugs**: 40-60 minutes
- **26 Chart bugs**: 1-2 hours
- **50 bugs (multi-project)**: 2-4 hours

## 💡 Tips

1. **Start small**: Collect 5 bugs first to test
2. **Use the quick script**: `quick_collect_bugs.sh` is easiest
3. **Monitor progress**: Check `multi_bug_data/` folder
4. **Combine gradually**: Test with 5, then add more
5. **Check disk space**: Each bug uses ~50-200 MB

## 🚀 Next Steps

1. ✅ Run `quick_collect_bugs.sh` in WSL
2. ✅ Wait for collection (~20-30 min)
3. ✅ Run `combine_multi_bug_data.py` in PowerShell
4. ✅ Update training script to use combined data
5. ✅ Retrain model - should see much better results!

## 📝 Notes

- **Class imbalance will still exist** but will be much better
- **Coverage matrix is simplified** (aggregate coverage, not per-test)
- **For production**, you'd want per-test coverage data (requires GZoltar or similar)
- **Current approach** is good enough for proof-of-concept and addressing imbalance

---

**Ready to start?** Run `bash quick_collect_bugs.sh` in WSL! 🎉

