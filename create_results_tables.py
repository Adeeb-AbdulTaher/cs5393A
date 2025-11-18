"""
Create results tables similar to the research paper
Based on available data from Chart-1 bug
"""
import numpy as np
import pandas as pd
import json

print("=" * 60)
print("Creating Results Tables")
print("=" * 60)

# Load data
X = np.load('line_coverage_matrix.npy')
y = np.load('line_coverage_labels.npy')
test_df = pd.read_csv('line_coverage_test_labels.csv')

# Load model results if available
try:
    predictions = np.load('predictions.npy')
    test_labels = np.load('test_labels.npy')
    has_model_results = True
except:
    has_model_results = False
    print("Note: Model predictions not found. Creating tables from coverage data only.")

# ============================================================================
# TABLE: Basic Results for Chart-1 (Single Bug)
# ============================================================================
print("\n1. Creating Basic Results Table...")

results_data = {
    'Metric': [
        'Total Tests',
        'Failing Tests',
        'Passing Tests',
        'Code Lines',
        'Coverage Rate',
        'Lines Covered',
        'Avg Lines per Test',
        'Avg Tests per Line'
    ],
    'Value': [
        len(y),
        int(y.sum()),
        int(len(y) - y.sum()),
        X.shape[1],
        f"{X.mean() * 100:.2f}%",
        int(X.sum()),
        f"{X.sum(axis=1).mean():.1f}",
        f"{X.sum(axis=0).mean():.1f}"
    ]
}

results_df = pd.DataFrame(results_data)
results_df.to_csv('table_basic_results.csv', index=False)
print("   ✓ Saved: table_basic_results.csv")
print("\n" + results_df.to_string(index=False))

# ============================================================================
# TABLE: Model Performance (if available)
# ============================================================================
if has_model_results:
    print("\n2. Creating Model Performance Table...")
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    
    model_perf = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"]
    })
    
    model_perf.to_csv('table_model_performance.csv', index=False)
    print("   ✓ Saved: table_model_performance.csv")
    print("\n" + model_perf.to_string(index=False))

# ============================================================================
# TABLE: Coverage Analysis by Test Type
# ============================================================================
print("\n3. Creating Coverage Analysis Table...")

failing_idx = np.where(y == 1)[0]
passing_idx = np.where(y == 0)[0]

if len(failing_idx) > 0:
    failing_coverage = X[failing_idx[0], :].sum()
    failing_avg = X[failing_idx[0], :].mean() * 100
else:
    failing_coverage = 0
    failing_avg = 0

passing_coverage = X[passing_idx, :].sum(axis=1)
passing_avg = passing_coverage.mean()
passing_avg_pct = X[passing_idx, :].mean() * 100

coverage_analysis = pd.DataFrame({
    'Test Type': ['Failing Tests', 'Passing Tests', 'Overall'],
    'Count': [len(failing_idx), len(passing_idx), len(y)],
    'Avg Lines Covered': [
        f"{failing_coverage:.1f}" if len(failing_idx) > 0 else "N/A",
        f"{passing_avg:.1f}",
        f"{X.sum(axis=1).mean():.1f}"
    ],
    'Coverage %': [
        f"{failing_avg:.2f}%" if len(failing_idx) > 0 else "N/A",
        f"{passing_avg_pct:.2f}%",
        f"{X.mean() * 100:.2f}%"
    ]
})

coverage_analysis.to_csv('table_coverage_analysis.csv', index=False)
print("   ✓ Saved: table_coverage_analysis.csv")
print("\n" + coverage_analysis.to_string(index=False))

# ============================================================================
# TABLE: What We CAN Create vs Paper Tables
# ============================================================================
print("\n4. Creating Comparison Table...")

comparison_data = {
    'Paper Table': [
        'TABLE III: Comparative Study',
        'TABLE IV: Ordering & StateDep',
        'TABLE V: Learning Representations',
        'TABLE VI: Cross-project vs Within-project',
        'TABLE VII: ManyBugs vs Defects4J'
    ],
    'Can Create?': [
        '❌ No - Need other methods (MULTRIC, FLUCCS, etc.)',
        '❌ No - Need Ordering/StateDep variants',
        '❌ No - Need mutation matrix, code rep, text sim',
        '⚠️ Partial - Can create for Chart project (but only 1 bug)',
        '❌ No - Need ManyBugs dataset and more Defects4J bugs'
    ],
    'What We Have': [
        'DEEPRL4FL baseline only',
        'Base version only',
        'Base (SpecMatrix) only',
        'Chart-1 bug only (need all Chart bugs)',
        '1 Defects4J bug (need 395+ bugs)'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('table_what_we_can_create.csv', index=False)
print("   ✓ Saved: table_what_we_can_create.csv")
print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# TABLE: Simplified Results (What We CAN Report)
# ============================================================================
print("\n5. Creating Simplified Results Table (What We Can Report)...")

# For a single bug, we can report basic metrics
simplified_results = pd.DataFrame({
    'Project': ['Chart'],
    'Bug ID': ['Chart-1'],
    'Total Tests': [len(y)],
    'Failing Tests': [int(y.sum())],
    'Code Lines': [X.shape[1]],
    'Coverage %': [f"{X.mean() * 100:.2f}%"],
    'Model Accuracy': [f"{accuracy_score(test_labels, predictions):.4f}" if has_model_results else "N/A"],
    'Notes': ['Single bug - limited for statistical analysis']
})

simplified_results.to_csv('table_simplified_results.csv', index=False)
print("   ✓ Saved: table_simplified_results.csv")
print("\n" + simplified_results.to_string(index=False))

print("\n" + "=" * 60)
print("✓ All tables created!")
print("=" * 60)
print("\nGenerated CSV files:")
print("  - table_basic_results.csv")
if has_model_results:
    print("  - table_model_performance.csv")
print("  - table_coverage_analysis.csv")
print("  - table_what_we_can_create.csv")
print("  - table_simplified_results.csv")

