#!/bin/bash

set -e

# Function to print messages
function print_message {
  echo "==================================="
  echo "$1"
  echo "==================================="
}

# Function to check if a command exists
function command_exists {
  command -v "$1" >/dev/null 2>&1
}

# Install prerequisites
print_message "Updating package lists..."
sudo apt-get update
print_message "Installing build-essential, wget, unzip, and git..."
sudo apt-get install -y build-essential wget unzip git

# Install Miniconda if not installed
if ! command_exists conda; then
  print_message "Installing Miniconda..."
  MINICONDA_INSTALLER_SCRIPT=Miniconda3-latest-Linux-x86_64.sh
  MINICONDA_PREFIX=/usr/local
  wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER_SCRIPT
  chmod +x $MINICONDA_INSTALLER_SCRIPT
  sudo ./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX
else
  print_message "Miniconda is already installed. Skipping installation."
fi

# Initialize Conda if not already initialized
if ! grep -q 'conda initialize' ~/.bashrc; then
  print_message "Initializing Conda..."
  export PATH=$MINICONDA_PREFIX/bin:$PATH
  conda init
  source ~/.bashrc
else
  print_message "Conda is already initialized. Skipping initialization."
  export PATH=$MINICONDA_PREFIX/bin:$PATH
fi

# Source Conda to ensure the environment is correctly set up
print_message "Sourcing Conda..."
if [ -f "$MINICONDA_PREFIX/etc/profile.d/conda.sh" ]; then
  source $MINICONDA_PREFIX/etc/profile.d/conda.sh
elif [ -f "/etc/profile.d/conda.sh" ]; then
  source /etc/profile.d/conda.sh
else
  print_message "Conda initialization script not found, trying alternative path..."
  source ~/.bashrc
fi

# Create a new Conda environment with Python 3.12.3 if it does not exist
if ! conda info --envs | grep -q 'py312'; then
  print_message "Creating Conda environment 'py312' with Python 3.12.3..."
  conda create -y -n py312 python=3.12.3
else
  print_message "Conda environment 'py312' already exists. Skipping creation."
fi

# Activate the Conda environment
print_message "Activating Conda environment 'py312'..."
source ~/.bashrc
conda activate py312

# Upgrade pip
print_message "Upgrading pip..."
pip install --upgrade pip

# Install CUDA 12.3 if not already installed
if ! dpkg-query -W -f='${Status}' cuda-toolkit-12-3 2>/dev/null | grep -q "install ok installed"; then
  print_message "Installing CUDA 12.3..."
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
  sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-wsl-ubuntu-12-3-local_12.3.0-1_amd64.deb
  sudo dpkg -i cuda-repo-wsl-ubuntu-12-3-local_12.3.0-1_amd64.deb
  sudo cp /var/cuda-repo-wsl-ubuntu-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
  sudo apt-get update
  sudo apt-get -y install cuda-toolkit-12-3
else
  print_message "CUDA 12.3 is already installed. Skipping installation."
fi

# Install cuDNN 8.9 if not already installed
if ! ls /usr/local/cuda/include/cudnn*.h 1> /dev/null 2>&1; then
  print_message "Installing cuDNN 8.9..."
  CUDNN_TAR_FILE="cudnn-12.3-linux-x64-v8.9.2.26.tgz"
  wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.9.2/$CUDNN_TAR_FILE
  tar -xzvf $CUDNN_TAR_FILE
  sudo cp -P cuda/include/cudnn*.h /usr/local/cuda/include
  sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
  sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
else
  print_message "cuDNN 8.9 is already installed. Skipping installation."
fi

# Export CUDA paths if not already exported
if ! grep -q 'export PATH=/usr/local/cuda/bin:$PATH' ~/.bashrc; then
  print_message "Exporting CUDA paths..."
  echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
  source ~/.bashrc
else
  print_message "CUDA paths are already exported. Skipping."
  source ~/.bashrc
fi

# Verify CUDA installation
print_message "Verifying CUDA installation..."
nvcc --version

# Install TensorFlow if not already installed
if ! pip show tensorflow | grep -q 'Version: 2.16.1'; then
  print_message "Installing TensorFlow..."
  pip install tensorflow==2.16.1
else
  print_message "TensorFlow 2.16.1 is already installed. Skipping installation."
fi

# Install other Python dependencies from requirements.txt
print_message "Installing additional Python dependencies..."
pip install -r requirements.txt

print_message "Installation complete. TensorFlow and all dependencies have been installed in the 'py312' environment."

# Clone the GitHub repository if not already cloned
REPO_URL="https://github.com/your-username/your-repository.git"
REPO_NAME=$(basename "$REPO_URL" .git)

if [ ! -d "$REPO_NAME" ]; then
  print_message "Cloning the GitHub repository..."
  git clone "$REPO_URL"
else
  print_message "GitHub repository already cloned. Skipping cloning."
fi

# Run the test script if it exists
if [ -f "./test_installation.sh" ]; then
  print_message "Running the test script..."
  ./test_installation.sh
else
  print_message "Test script not found. Skipping test run."
fi
