#!/bin/bash

# Activate the Python virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py312

# Set the PYTHONPATH to include the backend directory
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend

# Start the servers concurrently
npx concurrently --kill-others-on-fail "npm --prefix frontend/mood-prediction-app start" "python backend/server/main/server.py"
