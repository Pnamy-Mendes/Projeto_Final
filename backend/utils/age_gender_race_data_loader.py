import os
import numpy as np
import cv2
import logging
from sklearn.preprocessing import OneHotEncoder
from utils.helpers import load_config, extract_landmarks, print_memory_usage

def load_age_gender_race_data(data_dir):
    images = []
    ages = []
    genders = []
    races = []
    moods = []
    landmarks = []  # To store landmarks
    features = []  # To store extracted features

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            try:
                # Parse the filename to extract labels
                parts = filename.split('_')
                age = int(parts[0])
                gender = int(parts[1])
                if len(parts) > 2 and parts[2].isdigit():
                    race = int(parts[2])
                else:
                    logging.error(f"Filename {filename} does not contain a valid race part")
                    continue
                mood = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0  # Assuming mood label exists in filename
                file_path = os.path.join(data_dir, filename)

                # Load the image
                image = cv2.imread(file_path)
                image = cv2.resize(image, (256, 256))
                images.append(image)
                ages.append(age)
                genders.append(gender)
                races.append(race)
                moods.append(mood)

                # Extract landmarks and features
                landmark = extract_landmarks(image)
                landmarks.append(landmark)
                feature = extract_features(image, landmark)
                features.append(feature)

                if len(images) % 1000 == 0:
                    print_memory_usage()
                    logging.debug(f"Processed {len(images)} images")

            except Exception as e:
                logging.error(f"Error loading image {filename}: {e}")

    return np.array(images), np.array(ages), np.array(genders), np.array(races), np.array(moods), np.array(landmarks), np.array(features)

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

    # Additional features
    mouth_angle = calculate_angle(landmarks[48], landmarks[54])
    mouth_corner_angle_left = calculate_angle(landmarks[48], landmarks[66])
    mouth_corner_angle_right = calculate_angle(landmarks[54], landmarks[66])
    mouth_shape = mouth_shape_classification(mouth_ratio, mouth_corner_angle_left, mouth_corner_angle_right)
    mouth_corners_up = 1 if mouth_corner_angle_left > 0 and mouth_corner_angle_right < 0 else 0
    mouth_corners_down = 1 if mouth_corner_angle_left < 0 and mouth_corner_angle_right > 0 else 0

    additional_features = [
        mouth_height, mouth_width, mouth_ratio, eye_distance,
        eyebrow_distance, nose_length, eye_ratio,
        left_eye_ratio, right_eye_ratio,
        np.linalg.norm(landmarks[39] - landmarks[42]),  # Interocular distance
        np.linalg.norm(landmarks[31] - landmarks[35]),  # Nose width
        mouth_angle, mouth_corner_angle_left, mouth_corner_angle_right,
        mouth_corners_up, mouth_corners_down
    ]

    # Encode mouth shape as one-hot vector
    mouth_shape_dict = {'D': 0, 'O': 1, ')': 2, '(': 3, 'R': 4, '|': 5, 'unknown': 6}
    mouth_shape_one_hot = np.zeros(len(mouth_shape_dict))
    mouth_shape_one_hot[mouth_shape_dict[mouth_shape]] = 1

    features = additional_features + mouth_shape_one_hot.tolist()

    # Ensure features have the same length as expected by the model (pad if necessary)
    feature_length = 256  # Example feature length, update if needed
    if len(features) < feature_length:
        features += [0] * (feature_length - len(features))
    elif len(features) > feature_length:
        features = features[:feature_length]
    
    return np.array(features)

def calculate_angle(p1, p2):
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

def mouth_shape_classification(mouth_ratio, mouth_corner_angle_left, mouth_corner_angle_right):
    if mouth_ratio > 0.5:
        return 'D'  # Broad smile
    elif mouth_ratio > 0.3:
        return 'O'  # Open mouth (surprised)
    elif mouth_corner_angle_left > 20 and mouth_corner_angle_right < -20:
        return ')'  # Smile
    elif mouth_corner_angle_left < -20 and mouth_corner_angle_right > 20:
        return '('  # Sad
    elif mouth_corner_angle_left < -10 and mouth_corner_angle_right < -10:
        return 'R'  # Anger (teeth showing)
    elif mouth_ratio < 0.1:
        return '|'  # Neutral
    else:
        return 'unknown'
