# utils/feature_extraction.py
import os
import numpy as np
import cv2
import dlib
from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2gray
import logging

logging.basicConfig(level=logging.DEBUG)

def calculate_relative_distance(point1, point2, reference_distance):
    return np.linalg.norm(point1 - point2) / reference_distance

def detect_landmarks(image, predictor_path):
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            landmarks = np.zeros((68, 2), dtype="int")
            for i in range(0, 68):
                landmarks[i] = (shape.part(i).x, shape.part(i).y)
            return landmarks, rect
    except Exception as e:
        logging.error(f"Error detecting landmarks: {e}")
        return np.zeros((68, 2), dtype="int"), None
    return np.zeros((68, 2), dtype="int"), None

def extract_wrinkle_features(image, region):
    try:
        gray_region = rgb2gray(region)
        glcm = greycomatrix((gray_region * 255).astype(np.uint8), distances=[5], angles=[0], symmetric=True, normed=True)
        contrast = greycoprops(glcm, 'contrast')[0, 0]
        energy = greycoprops(glcm, 'energy')[0, 0]
        return np.array([contrast, energy])
    except Exception as e:
        logging.error(f"Error extracting wrinkle features: {e}")
        return np.array([0, 0])

def extract_additional_features(image, landmarks, face_rect):
    features = []

    # Calculate reference distances
    face_width = face_rect.width()
    face_height = face_rect.height()
    reference_distance = np.sqrt(face_width**2 + face_height**2)

    # Mouth features
    mouth_height = calculate_relative_distance(landmarks[62], landmarks[66], reference_distance)
    mouth_width = calculate_relative_distance(landmarks[60], landmarks[64], reference_distance)
    mouth_corner_left_angle = calculate_relative_distance(landmarks[48], landmarks[66], reference_distance)
    mouth_corner_right_angle = calculate_relative_distance(landmarks[54], landmarks[66], reference_distance)
    mouth_angle = np.degrees(np.arctan2(landmarks[66][1] - landmarks[62][1], landmarks[66][0] - landmarks[62][0]))

    features.extend([mouth_height, mouth_width, mouth_corner_left_angle, mouth_corner_right_angle, mouth_angle])

    # Eye features
    left_eye_width = calculate_relative_distance(landmarks[36], landmarks[39], reference_distance)
    left_eye_height = calculate_relative_distance(landmarks[37], landmarks[41], reference_distance)
    right_eye_width = calculate_relative_distance(landmarks[42], landmarks[45], reference_distance)
    right_eye_height = calculate_relative_distance(landmarks[43], landmarks[47], reference_distance)

    features.extend([left_eye_width, left_eye_height, right_eye_width, right_eye_height])

    # Wrinkle features around eyes
    left_eye_region = image[landmarks[37][1]-10:landmarks[41][1]+10, landmarks[36][0]-10:landmarks[39][0]+10]
    right_eye_region = image[landmarks[43][1]-10:landmarks[47][1]+10, landmarks[42][0]-10:landmarks[45][0]+10]
    left_eye_wrinkles = extract_wrinkle_features(image, left_eye_region)
    right_eye_wrinkles = extract_wrinkle_features(image, right_eye_region)

    features.extend(left_eye_wrinkles)
    features.extend(right_eye_wrinkles)

    # Additional features can be added here

    return np.array(features)

def extract_features(image_path, predictor_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        landmarks, face_rect = detect_landmarks(image, predictor_path)
        if face_rect is None:
            raise ValueError(f"No face detected in image {image_path}")
        
        additional_features = extract_additional_features(image, landmarks, face_rect)

        logging.debug(f"Extracted features for {image_path}: {additional_features}")
        return image, landmarks, additional_features
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return np.zeros((256, 256, 3), dtype=np.float32), np.zeros((68, 2), dtype=np.float32), np.zeros(20)
