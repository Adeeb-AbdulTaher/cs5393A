# Guide: Collecting More Data to Address Class Imbalance

## The Problem

With only Chart-1 bug, we have:
- **1 failing test** vs **2,192 passing tests** (0.046% failure rate)
- Extreme class imbalance makes it impossible for the model to learn

## The Solution

Collect data from **multiple Defects4J bugs** to get more failing tests.

## Available Data Sources

### 1. More Chart Bugs (Same Project)
Defects4J Chart project has **26 bugs**:
```bash
# In WSL, you can checkout:
Chart-1, Chart-2, Chart-3, ..., Chart-26
```

### 2. Other Defects4J Projects
Defects4J has multiple projects with many bugs:

| Project | Number of Bugs | Good for |
|---------|----------------|----------|
| **Chart** | 26 | Small, manageable |
| **Math** | 106 | Large dataset |
| **Time** | 26 | Medium size |
| **Closure** | 174 | Very large |
| **Lang** | 61 | Good size |
| **Mockito** | 38 | Medium size |
| **Cli** | 39 | Medium size |
| **Codec** | 18 | Small |
| **Collections** | 28 | Medium |
| **Compress** | 47 | Medium |
| **Csv** | 16 | Small |
| **Gson** | 18 | Small |
| **JacksonCore** | 26 | Medium |
| **JacksonDatabind** | 110 | Large |
| **JacksonXml** | 6 | Very small |
| **Jsoup** | 93 | Large |
| **JxPath** | 22 | Small |

**Total**: 854+ bugs available!

## How to Collect More Data

### Option 1: Manual Collection (Recommended to Start)

**Step 1: Checkout multiple bugs**
```bash
# In WSL terminal
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Checkout Chart bugs 1-5
for i in {1..5}; do
  sudo /root/defects4j/framework/bin/defects4j checkout -p Chart -v ${i}b -w /tmp/Chart_${i}b
done
```

**Step 2: Run tests and get coverage for each**
```bash
for i in {1..5}; do
  cd /tmp/Chart_${i}b
  sudo /root/defects4j/framework/bin/defects4j test
  sudo /root/defects4j/framework/bin/defects4j coverage
done
```

**Step 3: Copy data files to Windows**
```bash
# Copy all coverage files
for i in {1..5}; do
  mkdir -p /mnt/d/adeeb/Downloads/Shibbir\ Presentation/project/Chart_${i}
  cp /tmp/Chart_${i}b/coverage.xml /mnt/d/adeeb/Downloads/Shibbir\ Presentation/project/Chart_${i}/
  cp /tmp/Chart_${i}b/summary.csv /mnt/d/adeeb/Downloads/Shibbir\ Presentation/project/Chart_${i}/
  cp /tmp/Chart_${i}b/failing_tests /mnt/d/adeeb/Downloads/Shibbir\ Presentation/project/Chart_${i}/
  cp /tmp/Chart_${i}b/all_tests /mnt/d/adeeb/Downloads/Shibbir\ Presentation/project/Chart_${i}/
done
```

### Option 2: Automated Script

I've created `collect_multiple_bugs.py` to automate this:

```powershell
# Run the collection script
python collect_multiple_bugs.py
```

This will:
1. Checkout multiple bugs automatically
2. Run tests for each
3. Collect coverage data
4. Copy files to `multi_bug_data/` folder

### Option 3: Quick Start - Get 5 Chart Bugs

Run this in WSL:
```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Quick script to get Chart bugs 1-5
for i in 1 2 3 4 5; do
  echo "Processing Chart-$i..."
  sudo /root/defects4j/framework/bin/defects4j checkout -p Chart -v ${i}b -w /tmp/Chart_${i}b
  cd /tmp/Chart_${i}b
  sudo /root/defects4j/framework/bin/defects4j test
  sudo /root/defects4j/framework/bin/defects4j coverage
  echo "Chart-$i done"
done
```

## Expected Results

### With 5 Chart Bugs:
- **~5-10 failing tests** (instead of 1)
- **~10,000+ total tests** (instead of 2,193)
- **Class balance**: ~0.1-0.2% (still imbalanced but better)

### With 10 Chart Bugs:
- **~10-20 failing tests**
- **~20,000+ total tests**
- **Class balance**: ~0.1-0.2%

### With Multiple Projects (e.g., Chart + Math + Time):
- **50-100+ failing tests**
- **50,000+ total tests**
- **Class balance**: ~0.2-0.5% (still imbalanced, but much better)

## After Collecting Data

### Step 1: Combine Data
```powershell
python combine_multi_bug_data.py
```

This creates:
- `combined_coverage_matrix.npy` - All tests from all bugs
- `combined_test_labels.npy` - All labels
- `combined_test_labels.csv` - Human-readable format

### Step 2: Retrain Model
```python
# Load combined data
X = np.load('combined_coverage_matrix.npy')
y = np.load('combined_test_labels.npy')

# Retrain with better class balance
# (Update train_deeprl4fl.py to use combined data)
```

## Recommended Collection Strategy

### For Quick Results (1-2 hours):
- **Collect Chart bugs 1-10** (10 bugs)
- Expected: ~10-20 failing tests

### For Good Results (4-6 hours):
- **Collect Chart bugs 1-26** (all Chart bugs)
- Expected: ~26-50 failing tests

### For Best Results (1-2 days):
- **Collect from multiple projects**:
  - Chart: all 26 bugs
  - Math: bugs 1-20
  - Time: bugs 1-15
  - Lang: bugs 1-20
- Expected: **100+ failing tests**

## Time Estimates

- **Per bug**: ~2-5 minutes (checkout + tests + coverage)
- **10 bugs**: ~20-50 minutes
- **26 Chart bugs**: ~1-2 hours
- **50 bugs (multiple projects)**: ~2-4 hours

## Tips

1. **Start small**: Collect 5-10 bugs first to test the process
2. **Use automation**: The scripts will save you time
3. **Monitor disk space**: Each bug uses ~50-200 MB
4. **Check failures**: Some bugs might fail to checkout (deprecated bugs)
5. **Combine gradually**: Test with 5 bugs, then add more

## Next Steps

1. Run `collect_multiple_bugs.py` or use manual commands
2. Wait for collection to complete
3. Run `combine_multi_bug_data.py` to merge data
4. Retrain model with combined data
5. Compare results - should see much better failing test detection!

