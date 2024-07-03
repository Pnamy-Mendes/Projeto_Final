# utils/data_utils.py
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import dlib
from utils.feature_extraction import extract_features

def load_fer_data(data_dir, predictor_path, max_images_per_class=None):
    emotions = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    labels = []
    features_list = []

    for idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(data_dir, emotion)
        for i, file_name in enumerate(os.listdir(emotion_dir)):
            if max_images_per_class and i >= max_images_per_class:
                break
            file_path = os.path.join(emotion_dir, file_name)
            print(f"Processing image: {file_path}")

            _, _, features = extract_features(file_path, predictor_path)
            features_list.append(features)
            labels.append(idx)

    features = np.array(features_list)
    labels = np.array(labels)

    return features, labels
