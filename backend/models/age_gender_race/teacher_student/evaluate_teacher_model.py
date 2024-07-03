import sys
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
import cv2
import dlib
import logging

# Adjust the backend directory import
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(backend_dir)

try:
    from utils.age_gender_race_helpers import load_config, setup_tensorflow_gpu
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

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()
predictor_path = config['datasets']['predictor_path']
predictor = dlib.shape_predictor(predictor_path)

# Load the combined model
model_path = os.path.join(backend_dir, 'models', 'age_gender_race', 'models', 'teacher_model_best.keras')
combined_model = tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, "Image not found or unable to load"
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rects = detector(image_rgb, 0)
    if len(rects) == 0:
        return None, None, "No face detected"
    shape = predictor(image_rgb, rects[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    image_resized = cv2.resize(image_rgb, (256, 256))
    return image_resized, landmarks, None

def load_images_from_folder(folder, max_images=None):
    images = []
    labels = []
    for i, filename in enumerate(os.listdir(folder)):
        if max_images and i >= max_images:
            break
        if filename.endswith(".jpg"):
            try:
                age, gender, race, _ = filename.split("_")
                image_path = os.path.join(folder, filename)
                images.append(image_path)
                labels.append((int(age), int(gender), int(race)))
            except ValueError as e:
                print(f"Skipping file {filename}: {e}")
    return images, labels

def evaluate_model(model, image_paths, labels):
    age_predictions = []
    gender_predictions = []
    race_predictions = []
    valid_labels = []

    gender_count = np.zeros(2)
    gender_correct_count = np.zeros(2)
    race_count = np.zeros(5)
    race_correct_count = np.zeros(5)

    for image_path, label in zip(image_paths, labels):
        image, landmarks, error = preprocess_image(image_path)
        if error:
            print(f"Error processing {image_path}: {error}")
            continue
        if landmarks is None or len(landmarks) == 0 or np.any(landmarks == 0):
            print(f"Invalid landmarks for {image_path}. Skipping...")
            continue
        _, _, features = extract_features(image_path, config)
        features = np.expand_dims(features, axis=0)

        prediction = model.predict([np.expand_dims(image, axis=0), features])
        
        print(f"Predictions for {image_path}: Age={prediction[0][0][0]}, Gender={prediction[1][0]}, Race={prediction[2][0]}")
        
        age_predictions.append(prediction[0][0][0])
        gender_pred = np.argmax(prediction[1][0])
        race_pred = np.argmax(prediction[2][0])

        gender_count[gender_pred] += 1
        race_count[race_pred] += 1

        if gender_pred == label[1]:
            gender_correct_count[gender_pred] += 1
        if race_pred == label[2]:
            race_correct_count[race_pred] += 1

        gender_predictions.append(gender_pred)
        race_predictions.append(race_pred)
        valid_labels.append(label)

    valid_labels = np.array(valid_labels)
    y_val_age, y_val_gender, y_val_race = valid_labels[:, 0], valid_labels[:, 1], valid_labels[:, 2]

    # Calculate mean absolute error for age
    age_mae = mean_absolute_error(y_val_age, age_predictions)
    print(f"Mean Absolute Error for Age: {age_mae:.4f}")

    # Calculate accuracy for gender
    gender_accuracy = accuracy_score(y_val_gender, gender_predictions)
    print(f"Accuracy for Gender: {gender_accuracy:.4f}")

    # Calculate accuracy for race
    race_accuracy = accuracy_score(y_val_race, race_predictions)
    print(f"Accuracy for Race: {race_accuracy:.4f}")

    # Detailed accuracy for race
    race_report = classification_report(y_val_race, race_predictions, target_names=[f"Race {i}" for i in range(5)])
    print("\nClassification Report for Race:\n", race_report)

    # Detailed accuracy for gender
    gender_report = classification_report(y_val_gender, gender_predictions, target_names=['Male', 'Female'])
    print("\nClassification Report for Gender:\n", gender_report)

    # Confusion matrices
    print("\nConfusion Matrix for Race:\n", confusion_matrix(y_val_race, race_predictions))
    print("\nConfusion Matrix for Gender:\n", confusion_matrix(y_val_gender, gender_predictions))

    # Prediction counts
    print("\nGender Prediction Counts:")
    for i, count in enumerate(gender_count):
        print(f"Gender {i}: Predicted {int(count)} times, Correct {int(gender_correct_count[i])} times")

    print("\nRace Prediction Counts:")
    for i, count in enumerate(race_count):
        print(f"Race {i}: Predicted {int(count)} times, Correct {int(race_correct_count[i])} times")

    print("Evaluation completed.")

# Load images and labels
data_dir = config['datasets']['age_gender_race_data']
max_images = 100  # Limit the number of images to evaluate
image_paths, labels = load_images_from_folder(data_dir, max_images=max_images)

# Evaluate the model
evaluate_model(combined_model, image_paths, labels)
