#!/bin/bash

set -e

# Function to print messages
function print_message {
  echo "==================================="
  echo "$1"
  echo "==================================="
}

# Install prerequisites
print_message "Updating package lists..."
sudo apt-get update
print_message "Installing build-essential, wget, and unzip..."
sudo apt-get install -y build-essential wget unzip

# Install Miniconda
print_message "Installing Miniconda..."
MINICONDA_INSTALLER_SCRIPT=Miniconda3-latest-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
sudo ./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX

# Initialize Conda
print_message "Initializing Conda..."
export PATH=$MINICONDA_PREFIX/bin:$PATH
conda init

# Create a new Conda environment with Python 3.12.3
print_message "Creating Conda environment 'py312' with Python 3.12.3..."
conda create -y -n py312 python=3.12.3

# Activate the Conda environment
print_message "Activating Conda environment 'py312'..."
source ~/.bashrc
conda activate py312

# Upgrade pip
print_message "Upgrading pip..."
pip install --upgrade pip

# Install CUDA 12.3
print_message "Installing CUDA 12.3..."
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-wsl-ubuntu-12-3-local_12.3.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-3-local_12.3.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3

# Install cuDNN 8.9
print_message "Installing cuDNN 8.9..."
CUDNN_TAR_FILE="cudnn-12.3-linux-x64-v8.9.2.26.tgz"
wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.9.2/$CUDNN_TAR_FILE
tar -xzvf $CUDNN_TAR_FILE
sudo cp -P cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Export CUDA paths
print_message "Exporting CUDA paths..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA and cuDNN installation
print_message "Verifying CUDA installation..."
nvcc --version

# Install TensorFlow
print_message "Installing TensorFlow..."
pip install tensorflow==2.16.1

# Install other Python dependencies from requirements.txt
print_message "Installing additional Python dependencies..."
pip install -r requirements.txt

print_message "Installation complete. TensorFlow and all dependencies have been installed in the 'py312' environment."

# Run the test script
print_message "Running the test script..."
./test_installation.sh
