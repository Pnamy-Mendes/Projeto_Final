#!/bin/bash

# Create backend structure
mkdir -p backend
cd backend

mkdir -p server
mkdir -p server/main
mkdir -p server/templates

mkdir -p models
mkdir -p models/mood
mkdir -p models/age_gender_race

mkdir -p utils

mkdir -p static
mkdir -p static/results

# Create __init__.py files to make directories as packages
touch server/__init__.py
touch server/main/__init__.py
touch models/__init__.py
touch models/mood/__init__.py
touch models/age_gender_race/__init__.py
touch utils/__init__.py

# Create empty Python files
touch server/main/server.py
touch models/mood/mood_model.py
touch models/mood/attention_layer.py
touch models/age_gender_race/multi_output_model.py
touch models/age_gender_race/train_model.py
touch models/age_gender_race/predict_model.py
touch utils/helpers.py
touch utils/data_loader.py
touch utils/tensorflow_setup.py

# Create config.yaml
cat <<EOT > config.yaml
datasets:
  fer_train: dataset/FER/train
  fer_validation: dataset/FER/validation
  age_gender_race_data: dataset/UTKface_inthewild
  predictor_path: models/shape_predictor_68_face_landmarks.dat
logging:
  file: project.log
  level: DEBUG
model_params:
  dense_units_1: 256
  dense_units_2: 128
  dense_units_3: 64
  dense_units_4: 32
  dense_units_5: 16
  dropout_rate: 0.5
  input_shape_features:
  - 3
  input_shape_image:
  - 48
  - 48
  - 3
  leaky_relu_slope: 0.01
  n_classes: 7
  regularization_rate: 0.001
server:
  host: 0.0.0.0
  port: 5002
version: '12.3'
EOT

# Create requirements.txt
cat <<EOT > requirements.txt
Flask==2.0.1
Flask-Cors==3.0.10
tensorflow==2.16.1
dlib==19.22.0
numpy==1.21.2
opencv-python==4.5.3.56
PyYAML==5.4.1
EOT
