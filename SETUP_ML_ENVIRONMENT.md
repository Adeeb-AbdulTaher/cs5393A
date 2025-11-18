# ML Environment Setup Guide

## Install TensorFlow

TensorFlow installation can take several minutes. Run this command in your terminal:

```powershell
python -m pip install tensorflow
```

**Note**: If you're using a virtual environment (like `d4fl_env` mentioned in your instructions), activate it first:

```powershell
# Activate virtual environment (if you have one)
# d4fl_env\Scripts\activate  # Windows
# or
# source d4fl_env/bin/activate  # Linux/Mac

# Then install
pip install tensorflow
```

## Verify Installation

After installation, verify it works:

```powershell
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed!')"
```

## Alternative: Install PyTorch (if preferred)

If you prefer PyTorch instead:

```powershell
pip install torch torchvision torchaudio
```

## Run Training Script

Once TensorFlow is installed, run:

```powershell
python train_deeprl4fl.py
```

The script will:
1. Load the coverage matrices
2. Build a CNN model
3. Train on the data
4. Evaluate performance
5. Save the model and predictions

## Expected Output

You should see:
- Data loading confirmation
- Model architecture summary
- Training progress (epochs)
- Test accuracy and metrics
- Model saved to `deeprl4fl_model.h5`

## Troubleshooting

### If TensorFlow installation fails:
- Make sure you have Python 3.8-3.11 (TensorFlow may not support Python 3.12 yet)
- Try: `pip install tensorflow-cpu` (CPU-only version, smaller)
- Check your internet connection (large download ~500MB)

### If you get memory errors:
- Reduce batch_size in the script (change `batch_size=32` to `batch_size=16`)
- Reduce number of epochs

### If you want to use GPU:
- Install CUDA and cuDNN first
- Then install: `pip install tensorflow[and-cuda]`

