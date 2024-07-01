# backend/models/age_gender_race/teacher_student/evaluate_teacher_model.py
import sys
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import pandas as pd

# Adjust the backend directory import
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(backend_dir)

try:
    from utils.age_gender_race_helpers import load_config, setup_tensorflow_gpu, data_loader
    from utils.feature_extraction import extract_features
    print("Successfully imported modules.")
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    print("Make sure the utils directory contains the necessary modules.")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    exit(1)

# Load configuration
config = load_config(os.path.join(backend_dir, 'config.yaml'))

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load data
data_dir = config['datasets']['age_gender_race_data']
validation_split = config['validation_split']
input_shape = config['model_params']['input_shape']
use_cache_only = False  # Load data from cache

train_data, val_data = data_loader(data_dir, validation_split, input_shape, config, use_cache_only=use_cache_only)
x_val, y_val = val_data

y_val_age, y_val_gender, y_val_race = y_val

# Load the trained model
checkpoint_dir = os.path.join(backend_dir, 'models', 'age_gender_race', 'models')
model_path = os.path.join(checkpoint_dir, 'teacher_model_final.keras')
model = tf.keras.models.load_model(model_path)

# Make predictions
predictions = model.predict([x_val[0], x_val[1], x_val[2]])
pred_age, pred_gender, pred_race = predictions

# Convert predictions to appropriate formats
pred_age = np.squeeze(pred_age)
pred_gender = np.round(np.squeeze(pred_gender)).astype(int)
pred_race = np.argmax(pred_race, axis=1)

# Calculate mean absolute error for age
age_mae = mean_absolute_error(y_val_age, pred_age)
print(f"Mean Absolute Error for Age: {age_mae:.4f}")

# Calculate accuracy for gender
gender_accuracy = accuracy_score(y_val_gender, pred_gender)
print(f"Accuracy for Gender: {gender_accuracy:.4f}")

# Calculate accuracy for race
race_accuracy = accuracy_score(y_val_race, pred_race)
print(f"Accuracy for Race: {race_accuracy:.4f}")

# Detailed accuracy for race
race_report = classification_report(y_val_race, pred_race, target_names=[f"Race {i}" for i in range(5)])
print("\nClassification Report for Race:\n", race_report)

# Detailed accuracy for gender
gender_report = classification_report(y_val_gender, pred_gender, target_names=['Male', 'Female'])
print("\nClassification Report for Gender:\n", gender_report)

# Age group analysis
age_bins = [0, 18, 30, 45, 60, np.inf]
age_labels = ['0-17', '18-29', '30-44', '45-59', '60+']
y_val_age_group = pd.cut(y_val_age, bins=age_bins, labels=age_labels)
pred_age_group = pd.cut(pred_age, bins=age_bins, labels=age_labels)

age_group_accuracy = accuracy_score(y_val_age_group, pred_age_group)
print(f"Accuracy for Age Groups: {age_group_accuracy:.4f}")

age_group_report = classification_report(y_val_age_group, pred_age_group, target_names=age_labels)
print("\nClassification Report for Age Groups:\n", age_group_report)

# Function to calculate and print detailed accuracy for each class
def detailed_accuracy_per_class(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names)
    return report

# Detailed accuracy for race
race_class_names = [f"Race {i}" for i in range(5)]
print("\nDetailed Accuracy per Race Class:\n", detailed_accuracy_per_class(y_val_race, pred_race, race_class_names))

# Detailed accuracy for gender
gender_class_names = ['Male', 'Female']
print("\nDetailed Accuracy per Gender Class:\n", detailed_accuracy_per_class(y_val_gender, pred_gender, gender_class_names))

# Detailed accuracy for age groups
print("\nDetailed Accuracy per Age Group:\n", detailed_accuracy_per_class(y_val_age_group, pred_age_group, age_labels))

print("Evaluation completed.")
