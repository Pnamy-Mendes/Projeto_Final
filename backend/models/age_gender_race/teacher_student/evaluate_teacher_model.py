# backend/models/age_gender_race/teacher_student/evaluate_teacher_model.py
import sys
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
import cv2
import pandas as pd
import dlib
import keras_tuner as kt

# Adjust the backend directory import
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(backend_dir)

try:
    from utils.age_gender_race_helpers import load_config, setup_tensorflow_gpu
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

def build_model(hp):
    inputs = tf.keras.Input(shape=(196626,))
    x = tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu')(inputs)
    x = tf.keras.layers.Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1))(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Load the best model from Keras Tuner
tuner_dir = os.path.join(backend_dir, 'hyperband_teacher')
tuner = kt.Hyperband(
    hypermodel=build_model,  # Pass the model building function here
    objective='val_loss',
    max_epochs=10,
    factor=3,
    directory=tuner_dir,
    project_name='combined_tuning'
)
tuner.reload()
best_model = tuner.get_best_models(num_models=1)[0]

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if len(rects) == 0:
        return None, None, "No face detected"
    shape = predictor(gray, rects[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    image_resized = cv2.resize(image, (256, 256))
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

    for image_path, label in zip(image_paths, labels):
        image, landmarks, error = preprocess_image(image_path)
        if error:
            print(f"Error processing {image_path}: {error}")
            continue
        features = extract_features(image, landmarks)
        combined_features = np.concatenate([image.flatten(), landmarks.flatten(), features])
        combined_features = np.expand_dims(combined_features, axis=0)
        
        if combined_features.shape[1] < 196626:
            padding = np.zeros((combined_features.shape[0], 196626 - combined_features.shape[1]))
            combined_features = np.concatenate([combined_features, padding], axis=1)
        elif combined_features.shape[1] > 196626:
            combined_features = combined_features[:, :196626]
        
        prediction = model.predict(combined_features)
        age_predictions.append(prediction[0][0])
        gender_pred = np.round(prediction[0][1]).astype(int)
        gender_predictions.append(min(max(gender_pred, 0), 1))  # Constrain to 0 or 1
        race_predictions.append(np.argmax(prediction[0][2:]))
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

    print("Evaluation completed.")

def extract_features(image, landmarks):
    # Calculate features
    mouth_height = np.linalg.norm(landmarks[62] - landmarks[66])
    mouth_width = np.linalg.norm(landmarks[60] - landmarks[64])
    mouth_ratio = mouth_height / mouth_width if mouth_width != 0 else 0
    eye_distance = np.linalg.norm(landmarks[36] - landmarks[45])
    eyebrow_distance = np.linalg.norm(landmarks[19] - landmarks[24])
    nose_length = np.linalg.norm(landmarks[27] - landmarks[33])
    left_eye_ratio = np.linalg.norm(landmarks[37] - landmarks[41]) / np.linalg.norm(landmarks[36] - landmarks[39]) if np.linalg.norm(landmarks[36] - landmarks[39]) != 0 else 0
    right_eye_ratio = np.linalg.norm(landmarks[43] - landmarks[47]) / np.linalg.norm(landmarks[42] - landmarks[45]) if np.linalg.norm(landmarks[42] - landmarks[45]) != 0 else 0
    eye_ratio = (left_eye_ratio + right_eye_ratio) / 2

    additional_features = [
        mouth_height, mouth_width, mouth_ratio, eye_distance,
        eyebrow_distance, nose_length, eye_ratio,
        left_eye_ratio, right_eye_ratio,
        np.linalg.norm(landmarks[39] - landmarks[42]),  # Interocular distance
        np.linalg.norm(landmarks[31] - landmarks[35]),  # Nose width
    ]

    return np.array(additional_features)

# Load images and labels
data_dir = config['datasets']['age_gender_race_data']
max_images = 100  # Limit the number of images to evaluate
image_paths, labels = load_images_from_folder(data_dir, max_images=max_images)

# Evaluate the model
evaluate_model(best_model, image_paths, labels)
