"""Run SMOTE training with explicit output"""
import sys
import os

# Redirect stdout and stderr to file
log_file = open('smote_run.log', 'w')
sys.stdout = log_file
sys.stderr = log_file

try:
    # Import and run the training
    with open('train_smote_model.py', 'r', encoding='utf-8') as f:
        exec(f.read())
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print("Check smote_run.log for output")

