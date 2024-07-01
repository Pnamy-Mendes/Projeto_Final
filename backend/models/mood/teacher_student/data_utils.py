import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import dlib

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

def load_fer_data(data_dir, predictor_path, max_images_per_class=None):
    cnn_face_detector = dlib.cnn_face_detection_model_v1('./models/mmod_human_face_detector.dat')  # Use CNN model for GPU
    shape_predictor = dlib.shape_predictor(predictor_path)

    emotions = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    labels = []
    landmarks = []

    for idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(data_dir, emotion)
        for i, file_name in enumerate(os.listdir(emotion_dir)):
            if max_images_per_class and i >= max_images_per_class:
                break
            file_path = os.path.join(emotion_dir, file_name)
            print(f"Processing image: {file_path}")

            img = image.load_img(file_path, color_mode='grayscale')
            img_array = image.img_to_array(img).astype(np.uint8)

            # Convert grayscale image to 3-channel RGB
            if len(img_array.shape) == 2 or img_array.shape[2] == 1:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

            # Resize image to 48x48
            img_array = cv2.resize(img_array, (48, 48))

            # Detect face and landmarks using GPU
            dets = cnn_face_detector(img_array, 1)
            if len(dets) > 0:
                shape = shape_predictor(img_array, dets[0].rect)
                landmark = np.array([[p.x, p.y] for p in shape.parts()])

                # Calculate features
                mouth_height = np.linalg.norm(landmark[62] - landmark[66])
                mouth_width = np.linalg.norm(landmark[60] - landmark[64])
                mouth_ratio = mouth_height / mouth_width if mouth_width != 0 else 0
                eye_distance = np.linalg.norm(landmark[36] - landmark[45])
                eyebrow_distance = np.linalg.norm(landmark[19] - landmark[24])
                nose_length = np.linalg.norm(landmark[27] - landmark[33])
                left_eye_ratio = np.linalg.norm(landmark[37] - landmark[41]) / np.linalg.norm(landmark[36] - landmark[39]) if np.linalg.norm(landmark[36] - landmark[39]) != 0 else 0
                right_eye_ratio = np.linalg.norm(landmark[43] - landmark[47]) / np.linalg.norm(landmark[42] - landmark[45]) if np.linalg.norm(landmark[42] - landmark[45]) != 0 else 0
                eye_ratio = (left_eye_ratio + right_eye_ratio) / 2

                # Additional features
                mouth_angle = calculate_angle(landmark[48], landmark[54])
                mouth_corner_angle_left = calculate_angle(landmark[48], landmark[66])
                mouth_corner_angle_right = calculate_angle(landmark[54], landmark[66])
                mouth_shape = mouth_shape_classification(mouth_ratio, mouth_corner_angle_left, mouth_corner_angle_right)
                mouth_corners_up = 1 if mouth_corner_angle_left > 0 and mouth_corner_angle_right < 0 else 0
                mouth_corners_down = 1 if mouth_corner_angle_left < 0 and mouth_corner_angle_right > 0 else 0

                additional_features = [
                    mouth_height, mouth_width, mouth_ratio, eye_distance,
                    eyebrow_distance, nose_length, eye_ratio,
                    left_eye_ratio, right_eye_ratio,
                    np.linalg.norm(landmark[39] - landmark[42]),  # Interocular distance
                    np.linalg.norm(landmark[31] - landmark[35]),  # Nose width
                    mouth_angle, mouth_corner_angle_left, mouth_corner_angle_right,
                    mouth_corners_up, mouth_corners_down
                ]

                # Encode mouth shape as one-hot vector
                mouth_shape_dict = {'D': 0, 'O': 1, ')': 2, '(': 3, 'R': 4, '|': 5, 'unknown': 6}
                mouth_shape_one_hot = np.zeros(len(mouth_shape_dict))
                mouth_shape_one_hot[mouth_shape_dict[mouth_shape]] = 1

                features = additional_features + mouth_shape_one_hot.tolist()
                landmarks.append(features)
                labels.append(idx)
                print(f"Landmarks found for image: {file_path}")
            else:
                print(f"No landmarks detected for image: {file_path}. Skipping image.")

    features = np.array(landmarks)
    labels = np.array(labels)

    # Check for NaN values and handle them
    if np.any(np.isnan(features)):
        features = np.nan_to_num(features)

    return features, labels
