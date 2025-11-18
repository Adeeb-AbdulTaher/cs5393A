"""
Create coverage matrix for ML model input (DEEPRL4FL format)
Rows = test cases, Columns = code units (lines/methods)
Matrix[i,j] = 1 if test i covers line/method j, 0 otherwise
"""
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
import re

def parse_coverage_xml(filename='coverage.xml'):
    """Parse Cobertura coverage XML and extract line-level coverage"""
    tree = ET.parse(filename)
    root = tree.getroot()
    
    # Extract all covered lines with their hit counts
    line_coverage = {}
    classes_info = {}
    
    for package in root.findall('.//package'):
        pkg_name = package.get('name')
        for cls in package.findall('.//class'):
            class_name = cls.get('name')
            filename = cls.get('filename')
            full_class_name = f"{pkg_name}.{class_name}" if pkg_name else class_name
            
            classes_info[full_class_name] = {
                'filename': filename,
                'package': pkg_name,
                'class': class_name
            }
            
            # Extract line coverage
            for line in cls.findall('.//line'):
                line_num = int(line.get('number'))
                hits = int(line.get('hits', 0))
                line_key = f"{full_class_name}:{line_num}"
                line_coverage[line_key] = {
                    'class': full_class_name,
                    'line': line_num,
                    'hits': hits,
                    'filename': filename
                }
    
    return line_coverage, classes_info

def read_test_names(filename='all_tests'):
    """Read all test names"""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def extract_failing_test_names(filename='failing_tests'):
    """Extract failing test names from failing_tests file"""
    failing_tests = []
    with open(filename, 'r') as f:
        content = f.read()
        # Pattern: --- ClassName::testMethodName
        pattern = r'---\s+([\w\.]+::\w+)'
        matches = re.findall(pattern, content)
        failing_tests.extend(matches)
    return failing_tests

def create_coverage_matrix_simple(line_coverage, test_names, failing_tests):
    """
    Create a simplified coverage matrix.
    Note: Defects4J coverage.xml provides aggregate coverage, not per-test.
    This creates a structure that can be enhanced with per-test coverage data.
    """
    # Get all unique code units (lines)
    code_units = sorted(line_coverage.keys())
    num_tests = len(test_names)
    num_units = len(code_units)
    
    # Create matrix: rows = tests, cols = code units
    # For now, we'll mark lines as covered if they have hits > 0
    # In a full implementation, you'd need per-test coverage data
    matrix = np.zeros((num_tests, num_units), dtype=int)
    
    # Mark covered lines (simplified: if line is covered by any test)
    for j, unit in enumerate(code_units):
        if line_coverage[unit]['hits'] > 0:
            # Mark as covered by all tests (simplified approach)
            # In reality, you'd need per-test coverage data
            matrix[:, j] = 1
    
    # Create test labels (1 = failing, 0 = passing)
    test_labels = []
    for test in test_names:
        # Extract test method name from full test name
        # Format: testMethod(ClassName) -> ClassName::testMethod
        match = re.match(r'(\w+)\(([\w\.]+)\)', test)
        if match:
            test_method = match.group(1)
            test_class = match.group(2)
            test_key = f"{test_class}::{test_method}"
            label = 1 if test_key in failing_tests else 0
        else:
            label = 0
        test_labels.append(label)
    
    return matrix, code_units, test_labels

def create_method_level_matrix(classes_info, test_names, failing_tests):
    """Create method-level coverage matrix"""
    # Get all methods from classes
    methods = []
    method_to_index = {}
    
    for class_name, class_info in classes_info.items():
        # For method-level, we'll use class names as units
        # In a full implementation, parse methods from source code
        methods.append(class_name)
        method_to_index[class_name] = len(methods) - 1
    
    num_tests = len(test_names)
    num_methods = len(methods)
    
    # Create matrix (simplified: all methods covered)
    # In reality, you'd need per-test method coverage
    matrix = np.ones((num_tests, num_methods), dtype=int)
    
    # Create test labels
    test_labels = []
    for test in test_names:
        match = re.match(r'(\w+)\(([\w\.]+)\)', test)
        if match:
            test_method = match.group(1)
            test_class = match.group(2)
            test_key = f"{test_class}::{test_method}"
            label = 1 if test_key in failing_tests else 0
        else:
            label = 0
        test_labels.append(label)
    
    return matrix, methods, test_labels

def save_matrix_data(matrix, code_units, test_names, test_labels, prefix='coverage'):
    """Save matrix and metadata"""
    # Save matrix as numpy array
    np.save(f'{prefix}_matrix.npy', matrix)
    print(f"Saved matrix to {prefix}_matrix.npy: shape {matrix.shape}")
    
    # Save labels
    np.save(f'{prefix}_labels.npy', np.array(test_labels))
    print(f"Saved labels to {prefix}_labels.npy: {sum(test_labels)} failing, {len(test_labels) - sum(test_labels)} passing")
    
    # Save as CSV (may be large, so optional)
    if matrix.shape[1] < 10000:  # Only if not too large
        df = pd.DataFrame(matrix, index=test_names, columns=code_units)
        df.to_csv(f'{prefix}_matrix.csv')
        print(f"Saved CSV to {prefix}_matrix.csv")
    
    # Save metadata
    metadata = {
        'num_tests': len(test_names),
        'num_code_units': len(code_units),
        'num_failing': sum(test_labels),
        'num_passing': len(test_labels) - sum(test_labels),
        'code_units': code_units[:10]  # Sample
    }
    
    import json
    with open(f'{prefix}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {prefix}_metadata.json")
    
    # Save test names and labels
    test_df = pd.DataFrame({
        'test_name': test_names,
        'label': test_labels
    })
    test_df.to_csv(f'{prefix}_test_labels.csv', index=False)
    print(f"Saved test labels to {prefix}_test_labels.csv")

if __name__ == '__main__':
    print("Creating coverage matrix for ML model input...\n")
    
    # Step 1: Parse coverage data
    print("1. Parsing coverage.xml...")
    line_coverage, classes_info = parse_coverage_xml('coverage.xml')
    print(f"   Found {len(line_coverage)} covered lines")
    print(f"   Found {len(classes_info)} classes")
    
    # Step 2: Load test names
    print("\n2. Loading test names...")
    test_names = read_test_names('all_tests')
    print(f"   Found {len(test_names)} tests")
    
    # Step 3: Identify failing tests
    print("\n3. Identifying failing tests...")
    failing_tests = extract_failing_test_names('failing_tests')
    print(f"   Found {len(failing_tests)} failing tests: {failing_tests}")
    
    # Step 4: Create line-level matrix
    print("\n4. Creating line-level coverage matrix...")
    line_matrix, code_units, test_labels = create_coverage_matrix_simple(
        line_coverage, test_names, failing_tests
    )
    print(f"   Matrix shape: {line_matrix.shape} (tests x lines)")
    print(f"   Coverage density: {line_matrix.sum() / line_matrix.size * 100:.2f}%")
    
    # Step 5: Create method-level matrix
    print("\n5. Creating method-level coverage matrix...")
    method_matrix, methods, method_labels = create_method_level_matrix(
        classes_info, test_names, failing_tests
    )
    print(f"   Matrix shape: {method_matrix.shape} (tests x methods)")
    
    # Step 6: Save data
    print("\n6. Saving matrices...")
    save_matrix_data(line_matrix, code_units, test_names, test_labels, 'line_coverage')
    save_matrix_data(method_matrix, methods, test_names, method_labels, 'method_coverage')
    
    print("\n✓ Coverage matrices created successfully!")
    print("\nFiles created:")
    print("  - line_coverage_matrix.npy: Line-level coverage matrix")
    print("  - line_coverage_labels.npy: Test labels (1=failing, 0=passing)")
    print("  - method_coverage_matrix.npy: Method-level coverage matrix")
    print("  - method_coverage_labels.npy: Test labels")
    print("  - *_test_labels.csv: Test names with labels")
    print("  - *_metadata.json: Matrix metadata")
    
    print("\nNote: This is a simplified version using aggregate coverage.")
    print("For per-test coverage, you may need to run Defects4J with per-test")
    print("coverage collection or use additional tools.")

