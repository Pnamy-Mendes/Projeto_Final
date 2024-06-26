#!/bin/bash

# Activate conda environment
source /home/pnamy/anaconda3/etc/profile.d/conda.sh
conda activate py312

echo "Starting backend server..."
python backend/server/main/server.py &

# Ensure we're in the correct directory for the frontend server
cd frontend/mood-prediction-app

echo "Starting frontend server..."
npm start
