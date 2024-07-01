#!/bin/bash

set -e

# Function to print messages
function print_message {
  echo "==================================="
  echo "$1"
  echo "==================================="
}

# Activate the Conda environment
print_message "Activating Conda environment 'py312'..."
source ~/.bashrc
conda activate py312

# Test TensorFlow GPU availability
print_message "Testing TensorFlow GPU availability..."
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Print CUDA version
print_message "Printing CUDA version..."
nvcc --version

# Print cuDNN version
print_message "Printing cuDNN version..."
python -c "import tensorflow as tf; print('cuDNN version:', tf.sysconfig.get_build_info()['cuda_version'])"

print_message "Printing NVIDIA toolkit version..."
nvidia-smi

print_message "All tests completed successfully."
