# Chart-1b Data Summary

## Files Available

All files are now in your project directory: `D:\adeeb\Downloads\Shibbir Presentation\project`

### 1. `coverage.xml` (111 KB)
- **Format**: Cobertura XML coverage report
- **Content**: Line-by-line and method-level coverage data
- **Coverage Stats**:
  - Line coverage: 55.4% (318/574 lines)
  - Condition coverage: 46.3% (112/242 conditions)
- **Use**: Parse with XML parser (Python: `xml.etree.ElementTree` or `lxml`)

### 2. `summary.csv` (74 bytes)
- **Format**: CSV with header row
- **Content**: Overall coverage statistics
- **Columns**: `LinesTotal`, `LinesCovered`, `ConditionsTotal`, `ConditionsCovered`
- **Values**: 574, 318, 242, 112
- **Use**: Quick stats, easy to load with pandas: `pd.read_csv('summary.csv')`

### 3. `failing_tests` (3 KB)
- **Format**: Text file with test failure details
- **Content**: 
  - Test name: `org.jfree.chart.renderer.category.junit.AbstractCategoryItemRendererTests::test2947660`
  - Error: `AssertionFailedError: expected:<1> but was:<0>`
  - Full stack trace
- **Use**: Identify which tests fail and why

### 4. `all_tests` (150 KB)
- **Format**: Text file, one test per line
- **Content**: List of all 2,193 executed test methods
- **Format**: `testMethodName(ClassName)`
- **Use**: Get list of all tests, calculate test metrics

## Quick Python Examples

### Load Summary CSV
```python
import pandas as pd
df = pd.read_csv('summary.csv')
print(df)
```

### Parse Coverage XML
```python
import xml.etree.ElementTree as ET
tree = ET.parse('coverage.xml')
root = tree.getroot()
# Extract coverage data
```

### Read All Tests
```python
with open('all_tests', 'r') as f:
    tests = [line.strip() for line in f]
print(f"Total tests: {len(tests)}")
```

### Read Failing Tests
```python
with open('failing_tests', 'r') as f:
    failing = f.read()
print(failing)
```

## Bug Information

- **Bug ID**: Chart-1
- **Buggy Version**: 1b (revision 2264)
- **Fixed Version**: 1f (revision 2266)
- **Root Cause**: `org.jfree.chart.renderer.category.AbstractCategoryItemRenderer`
- **Failing Test**: `AbstractCategoryItemRendererTests::test2947660`
- **Error**: Expected 1 but got 0

## Next Steps for ML/Analysis

1. **Parse coverage.xml** to extract line-by-line coverage
2. **Map coverage to source code** to identify covered/uncovered lines
3. **Analyze test patterns** from all_tests
4. **Compare buggy vs fixed** versions (you have 1b, can checkout 1f)
5. **Extract features** for machine learning models

