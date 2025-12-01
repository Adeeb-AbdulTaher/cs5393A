"""
Run SMOTE training and save output to file
"""
import sys
import subprocess

# Run SMOTE training and capture output
result = subprocess.run(
    [sys.executable, 'train_smote_model.py'],
    capture_output=True,
    text=True,
    cwd='.'
)

# Print output
print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")

# Save to file
with open('smote_training_output.txt', 'w') as f:
    f.write("STDOUT:\n")
    f.write(result.stdout)
    f.write("\nSTDERR:\n")
    f.write(result.stderr)

print("\nOutput saved to smote_training_output.txt")

