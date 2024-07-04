import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import dlib
import cv2
import yaml
from sklearn.metrics import confusion_matrix, classification_report

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define paths
val_data_dir = config['datasets']['fer_validation']
predictor_path = config['datasets']['predictor_path']
teacher_model_path = 'models/mood/teacher_model_with_features.keras'

# Function to calculate relative features
def calculate_relative_features(landmarks):
    if len(landmarks) == 0:
        return np.zeros(12)
    try:
        face_width = np.linalg.norm(landmarks[36] - landmarks[45])
        mouth_height = np.linalg.norm(landmarks[62] - landmarks[66]) / face_width
        mouth_width = np.linalg.norm(landmarks[60] - landmarks[64]) / face_width
        eye_distance = np.linalg.norm(landmarks[36] - landmarks[45]) / face_width
        eyebrow_distance = np.linalg.norm(landmarks[19] - landmarks[24]) / face_width
        nose_length = np.linalg.norm(landmarks[27] - landmarks[33]) / face_width
        left_eye_ratio = np.linalg.norm(landmarks[37] - landmarks[41]) / np.linalg.norm(landmarks[36] - landmarks[39])
        right_eye_ratio = np.linalg.norm(landmarks[43] - landmarks[47]) / np.linalg.norm(landmarks[42] - landmarks[45])
        eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
        mouth_angle = np.degrees(np.arctan2(landmarks[54][1] - landmarks[48][1], landmarks[54][0] - landmarks[48][0]))
        inner_lip_distance = np.linalg.norm(landmarks[62] - landmarks[66]) / face_width
        outer_lip_corners_up = 1 if landmarks[54][1] < landmarks[48][1] else 0
        outer_lip_corners_down = 1 if landmarks[54][1] > landmarks[48][1] else 0

        features = np.array([
            mouth_height, mouth_width, eye_distance, eyebrow_distance, nose_length, eye_ratio,
            left_eye_ratio, right_eye_ratio, mouth_angle, inner_lip_distance,
            outer_lip_corners_up, outer_lip_corners_down
        ])
        return features
    except Exception as e:
        print(f"Error calculating relative features: {e}")
        return np.zeros(12)

# Function to extract features from images
def extract_features(image_path, predictor_path):
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        rects = detector(image, 1)
        if len(rects) > 0:
            shape = predictor(image, rects[0])
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            relative_features = calculate_relative_features(landmarks)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            resized_image = cv2.resize(image_rgb, (48, 48))
            return resized_image, relative_features
        else:
            return np.zeros((48, 48, 3)), np.zeros(12)
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return np.zeros((48, 48, 3)), np.zeros(12)

# Function to load data and extract features with a max_images_per_class parameter
def load_data(data_dir, predictor_path, subset, batch_size=64, max_images_per_class=10):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset=subset,
        shuffle=False
    )

    images = []
    labels = []
    features = []
    class_counts = {}

    print(f"Processing {subset} data...")
    for filename in generator.filenames:
        class_label = filename.split('/')[0]
        if max_images_per_class is not None:
            if class_label not in class_counts:
                class_counts[class_label] = 0
            if class_counts[class_label] >= max_images_per_class:
                continue
            class_counts[class_label] += 1

        image_path = os.path.join(data_dir, filename)
        image, feature = extract_features(image_path, predictor_path)

        if image is not None and feature is not None and feature.shape == (12,):
            images.append(image)
            features.append(feature)
            label = generator.class_indices[class_label]
            labels.append(label)
        else:
            print(f"Skipping invalid data for image: {image_path}")

    return np.array(images), np.array(features), to_categorical(labels, num_classes=7), generator.class_indices

# Load the trained model
model = load_model(teacher_model_path)

# Load validation data
val_images, val_features, val_labels, class_indices = load_data(val_data_dir, predictor_path, 'validation', max_images_per_class=10)

# Make predictions on the validation data
predictions = model.predict([val_images, val_features])

# Convert predictions and true labels to class labels
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(val_labels, axis=1)

# Generate a confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Generate a classification report
class_report = classification_report(true_labels, predicted_labels, target_names=class_indices.keys())

# Print detailed evaluation metrics
print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

# Calculate the number of correct and incorrect predictions for each class
correct_predictions = (predicted_labels == true_labels).sum()
total_predictions = len(true_labels)

print(f"\nTotal Correct Predictions: {correct_predictions}")
print(f"Total Predictions: {total_predictions}")
print(f"Accuracy: {correct_predictions / total_predictions:.2f}")

# Print number of times each class was predicted and how many of those predictions were correct
class_names = class_indices.keys()
for idx, class_name in enumerate(class_names):
    class_correct = ((predicted_labels == idx) & (true_labels == idx)).sum()
    class_total = (predicted_labels == idx).sum()
    print(f"\nClass '{class_name}':")
    print(f"  Total Predictions: {class_total}")
    print(f"  Correct Predictions: {class_correct}")
    print(f"  Accuracy: {class_correct / class_total if class_total > 0 else 0:.2f}")
