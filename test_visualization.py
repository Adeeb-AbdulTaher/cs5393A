"""Test visualization setup"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

print("Testing visualization...")
try:
    X = np.load('line_coverage_matrix.npy')
    print(f"Loaded matrix: {X.shape}")
    
    # Create simple test plot
    plt.figure(figsize=(10, 8))
    plt.imshow(X[:100, :100], cmap='Greys', aspect='auto')
    plt.title('Test Coverage Matrix')
    plt.xlabel('Code Lines')
    plt.ylabel('Test Cases')
    plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Test plot created: test_plot.png")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

