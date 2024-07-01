import cv2
import numpy as np
import logging
import tensorflow as tf
from keras.models import load_model
from utils.feature_extraction import detect_landmarks, extract_hair_color, extract_facial_hair, extract_face_structure

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the models
model_path = 'models/age_gender_race_model.keras'
model = load_model(model_path)

# Labels
age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
gender_ranges = ['male', 'female']
race_ranges = ['white', 'black', 'asian', 'indian', 'others']
mood_ranges = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def process_image(image_path, predictor_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    landmarks = detect_landmarks(image, predictor_path)
    hair_color = extract_hair_color(image, landmarks)
    facial_hair_density, facial_hair_color = extract_facial_hair(image, landmarks)
    face_structure = extract_face_structure(landmarks)

    additional_features = np.concatenate([hair_color, [facial_hair_density], facial_hair_color, face_structure], axis=0)

    return image_resized, landmarks, additional_features

def predict(image_path, predictor_path):
    try:
        image, landmarks, features = process_image(image_path, predictor_path)
        image = np.expand_dims(image, axis=0)
        landmarks = np.expand_dims(landmarks, axis=0)
        features = np.expand_dims(features, axis=0)

        predictions = model.predict([image, landmarks, features])

        mood_prediction = mood_ranges[np.argmax(predictions[0])]
        age_prediction = predictions[1][0][0]
        gender_prediction = gender_ranges[int(np.round(predictions[2][0][0]))]
        race_prediction = race_ranges[np.argmax(predictions[3])]

        return {
            "mood": mood_prediction,
            "age": age_prediction,
            "gender": gender_prediction,
            "race": race_prediction
        }
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None

if __name__ == "__main__":
    predictor_path = 'models/shape_predictor_68_face_landmarks.dat'
    test_image_path = 'path_to_test_image.jpg'
    
    result = predict(test_image_path, predictor_path)
    if result:
        print(f"Predictions: {result}")
