"""
Simple Python script to parse Defects4J coverage data
"""
import xml.etree.ElementTree as ET
import pandas as pd
import json

def parse_coverage_xml(filename='coverage.xml'):
    """Parse Cobertura coverage XML file"""
    tree = ET.parse(filename)
    root = tree.getroot()
    
    coverage_data = {
        'line_rate': float(root.get('line-rate')),
        'branch_rate': float(root.get('branch-rate')),
        'lines_covered': int(root.get('lines-covered')),
        'lines_valid': int(root.get('lines-valid')),
        'branches_covered': int(root.get('branches-covered')),
        'branches_valid': int(root.get('branches-valid')),
    }
    
    # Extract class-level coverage
    classes = []
    for package in root.findall('.//package'):
        for cls in package.findall('.//class'):
            class_info = {
                'package': package.get('name'),
                'class': cls.get('name'),
                'filename': cls.get('filename'),
                'line_rate': float(cls.get('line-rate')),
                'branch_rate': float(cls.get('branch-rate')),
            }
            classes.append(class_info)
    
    return coverage_data, classes

def parse_summary_csv(filename='summary.csv'):
    """Parse summary CSV file"""
    return pd.read_csv(filename)

def read_all_tests(filename='all_tests'):
    """Read all test names"""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def read_failing_tests(filename='failing_tests'):
    """Read failing test details"""
    with open(filename, 'r') as f:
        return f.read()

if __name__ == '__main__':
    print("Parsing Defects4J coverage data...\n")
    
    # Parse coverage XML
    coverage_data, classes = parse_coverage_xml()
    print("Coverage Summary:")
    print(f"  Line Coverage: {coverage_data['line_rate']*100:.2f}%")
    print(f"  Branch Coverage: {coverage_data['branch_rate']*100:.2f}%")
    print(f"  Lines: {coverage_data['lines_covered']}/{coverage_data['lines_valid']}")
    print(f"  Branches: {coverage_data['branches_covered']}/{coverage_data['branches_valid']}")
    
    # Parse summary CSV
    summary = parse_summary_csv()
    print("\nSummary CSV:")
    print(summary)
    
    # Read all tests
    all_tests = read_all_tests()
    print(f"\nTotal Tests: {len(all_tests)}")
    print(f"Sample tests: {all_tests[:5]}")
    
    # Read failing tests
    failing = read_failing_tests()
    print(f"\nFailing Tests:\n{failing[:500]}...")  # First 500 chars
    
    # Save parsed data
    classes_df = pd.DataFrame(classes)
    classes_df.to_csv('class_coverage.csv', index=False)
    print("\nClass-level coverage saved to 'class_coverage.csv'")

