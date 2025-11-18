"""Check if ML environment is ready"""
import sys

print("=" * 60)
print("ML Environment Check")
print("=" * 60)

# Check Python version
print(f"\n1. Python Version: {sys.version}")
python_version = sys.version_info
if python_version.major == 3 and 8 <= python_version.minor <= 11:
    print("   ✓ Python version compatible with TensorFlow")
else:
    print(f"   ⚠️  Python {python_version.major}.{python_version.minor} - TensorFlow may not support Python 3.12+")
    print("   Consider using Python 3.8-3.11")

# Check TensorFlow
print("\n2. TensorFlow:")
try:
    import tensorflow as tf
    print(f"   ✓ TensorFlow {tf.__version__} installed")
    print(f"   GPU devices: {len(tf.config.list_physical_devices('GPU'))}")
except ImportError:
    print("   ✗ TensorFlow not installed")
    print("   Run: pip install tensorflow")

# Check NumPy
print("\n3. NumPy:")
try:
    import numpy as np
    print(f"   ✓ NumPy {np.__version__} installed")
except ImportError:
    print("   ✗ NumPy not installed")
    print("   Run: pip install numpy")

# Check scikit-learn
print("\n4. scikit-learn:")
try:
    import sklearn
    print(f"   ✓ scikit-learn {sklearn.__version__} installed")
except ImportError:
    print("   ✗ scikit-learn not installed")
    print("   Run: pip install scikit-learn")

# Check required files
print("\n5. Required Data Files:")
import os
files_needed = ['line_coverage_matrix.npy', 'line_coverage_labels.npy']
all_present = True
for f in files_needed:
    if os.path.exists(f):
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"   ✓ {f} ({size:.2f} MB)")
    else:
        print(f"   ✗ {f} - MISSING")
        all_present = False

print("\n" + "=" * 60)
if all_present:
    try:
        import tensorflow as tf
        print("✓ Environment is ready! You can run: python train_deeprl4fl.py")
    except ImportError:
        print("⚠️  Install TensorFlow first: pip install tensorflow")
else:
    print("✗ Missing required data files. Run create_coverage_matrix.py first")
print("=" * 60)

