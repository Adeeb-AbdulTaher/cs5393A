"""Install and verify TensorFlow"""
import subprocess
import sys

print("Installing TensorFlow...")
result = subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow"], 
                       capture_output=True, text=True)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)

print("\nVerifying installation...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} installed successfully!")
    print(f"  GPU available: {tf.config.list_physical_devices('GPU')}")
except ImportError as e:
    print(f"✗ TensorFlow not found: {e}")

