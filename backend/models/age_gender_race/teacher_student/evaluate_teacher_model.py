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
batch_size = config['model_params']['batch_size']
max_images = 1000  # Limit the number of images to avoid memory issues
use_cache_only = True  # Load only cached images

try:
    train_gen, val_gen, steps_per_epoch_train, steps_per_epoch_val = data_loader(
        data_dir, validation_split, input_shape, config, batch_size, max_images=max_images, use_cache_only=use_cache_only
    )
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

def combine_data(generator):
    combined_inputs_list = []
    combined_targets_list = []
    for batch_idx, batch in enumerate(generator):
        print(f"Processing batch {batch_idx}")
        print(f"Batch contents: {batch}")
        inputs, targets = batch
        print(f"Inputs: {inputs}")
        print(f"Targets: {targets}")
        if isinstance(inputs, (list, tuple)):
            combined_inputs = tf.concat([tf.reshape(inputs[i], [inputs[i].shape[0], -1]) for i in range(len(inputs))], axis=1)
        else:
            combined_inputs = tf.reshape(inputs, [inputs.shape[0], -1])
        
        if isinstance(targets, (list, tuple)):
            combined_targets = tf.concat([tf.reshape(targets[i], [targets[i].shape[0], -1]) for i in range(len(targets))], axis=1)
        else:
            combined_targets = tf.reshape(targets, [targets.shape[0], -1])
        
        combined_inputs_list.append(combined_inputs)
        combined_targets_list.append(combined_targets)
        
    return tf.concat(combined_inputs_list, axis=0), tf.concat(combined_targets_list, axis=0)

# Convert generator to list to handle it properly
val_data_combined = list(val_gen)
x_val_combined, y_val_combined = combine_data(val_data_combined)

# Split y_val_combined into separate labels
y_val_age = y_val_combined[:, 0]
y_val_gender = y_val_combined[:, 1]
y_val_race = y_val_combined[:, 2:]

# Load the trained model
checkpoint_dir = os.path.join(backend_dir, 'models', 'age_gender_race', 'models')
model_path = os.path.join(checkpoint_dir, 'teacher_model_final.keras')
model = tf.keras.models.load_model(model_path)

# Make predictions
predictions = model.predict(x_val_combined)
pred_age = predictions[:, 0]
pred_gender = np.round(predictions[:, 1]).astype(int)
pred_race = np.argmax(predictions[:, 2:], axis=1)

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
