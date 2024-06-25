#!/bin/bash

# Navigate to frontend and start React app
cd frontend
echo "Starting React app..."
gnome-terminal -- bash -c "npm start; exec bash"

# Navigate to backend and start Flask server
cd ../backend
echo "Starting Flask server..."
gnome-terminal -- bash -c "python3 server/main/server.py; exec bash"

echo "Both React app and Flask server are starting..."
