from flask import Flask, render_template, request, jsonify
import os
import cv2
import dlib
import base64
import numpy as np
import tensorflow as tf
import yaml
import json
from datetime import datetime
import sys
from flask_cors import CORS
import socket

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.mood.attention_layer import add_attention_layer

# Define the template and static folder paths
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../frontend/templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../frontend/static'))

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Load configuration
config_path = 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Ensure model directory exists
mood_model_dir = "./models/mood/models"
age_gender_race_model_dir = "./models/age_gender_race/models"
if not os.path.exists(mood_model_dir):
    os.makedirs(mood_model_dir)

# Load the trained mood model
mood_model_path = os.path.join(mood_model_dir, 'teacher_model.keras')
mood_model = tf.keras.models.load_model(mood_model_path)

# Load the trained age, gender, race model
age_gender_race_model_path = os.path.join(age_gender_race_model_dir, 'teacher_model_best.keras')
age_gender_race_model = tf.keras.models.load_model(age_gender_race_model_path)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()
predictor_path = config['datasets']['predictor_path']
predictor = dlib.shape_predictor(predictor_path)

mood_label_map = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}

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

def preprocess_image(image_data):
    try:
        image = base64.b64decode(image_data.split(",")[1])
        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if len(rects) == 0:
            return None, None, None, "No face detected"
        shape = predictor(gray, rects[0])
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        image_resized = cv2.resize(image, (48, 48))
        return image, image_resized, landmarks, None
    except Exception as e:
        return None, None, None, f"Preprocessing error: {e}"

def draw_landmarks(image, landmarks):
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_picture')
def load_picture():
    return render_template('load_picture.html')

@app.route('/train_model')
def train_model():
    return render_template('train_model.html')

@app.route('/predict_mood', methods=['POST'])
def predict_mood():
    try:
        data = request.json
        image_data = data['image']
        original_image, resized_image, landmarks, error = preprocess_image(image_data)
        if error:
            print(f"Preprocessing error: {error}")
            return jsonify({'error': error})

        # Extract features from the resized image and landmarks
        features = extract_features(resized_image, landmarks)
        features = np.expand_dims(features.astype('float32') / 255.0, axis=0)

        # Debug: Print the shapes of image and features
        print(f"Shape of resized_image: {resized_image.shape}")
        print(f"Shape of landmarks: {landmarks.shape}")
        print(f"Shape of features: {features.shape}")

        # Predict with the mood model
        image_expanded = np.expand_dims(resized_image.astype('float32') / 255.0, axis=0)
        mood_prediction = mood_model.predict(image_expanded)
        mood_idx = np.argmax(mood_prediction[0])
        mood = mood_label_map[mood_idx]
        mood_confidence = int(np.max(mood_prediction[0]) * 100)

        # Prepare combined inputs for the age_gender_race model
        image_flattened = resized_image.flatten()  # Correctly flatten the resized image
        landmarks_flattened = landmarks.flatten()
        combined_features = np.concatenate([
            image_flattened,
            landmarks_flattened,
            features.flatten()
        ])
        combined_features = np.expand_dims(combined_features, axis=0)

        # Debug: Print the shape of combined_features
        print(f"Shape of combined_features: {combined_features.shape}")

        # Predict with the age, gender, race model
        age_gender_race_prediction = age_gender_race_model.predict(combined_features)
        pred_age = age_gender_race_prediction[:, 0]
        pred_gender = 'Male' if age_gender_race_prediction[:, 1] > 0.5 else 'Female'
        pred_race_idx = np.argmax(age_gender_race_prediction[:, 2:], axis=1)
        race_label_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}
        pred_race = race_label_map[pred_race_idx[0]]

        # Debug: Print predictions
        print(f"Mood Prediction: {mood_prediction[0]}, Mood: {mood}, Mood Confidence: {mood_confidence}")
        print(f"Age, Gender, Race Prediction: {age_gender_race_prediction[0]}, Age: {pred_age}, Gender: {pred_gender}, Race: {pred_race}")

        # Draw landmarks on the original image
        image_with_landmarks = draw_landmarks(original_image.copy(), landmarks)

        # Save the image with landmarks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_image_path = os.path.join(app.static_folder, 'results', f"result_{timestamp}.png")
        cv2.imwrite(result_image_path, image_with_landmarks)

        # Save the prediction data
        history_path = os.path.join(app.static_folder, 'results', 'history.json')
        history_data = []
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history_data = json.load(f)
        history_data.append({
            'timestamp': timestamp,
            'mood': mood,
            'mood_confidence': mood_confidence,
            'age': float(pred_age),
            'gender': pred_gender,
            'race': pred_race,
            'image_path': f"results/result_{timestamp}.png"
        })
        with open(history_path, 'w') as f:
            json.dump(history_data, f)

        return jsonify({
            'mood': mood,
            'mood_confidence': mood_confidence,
            'age': float(pred_age),
            'gender': pred_gender,
            'race': pred_race,
            'image_path': f"/static/results/result_{timestamp}.png"
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/history', methods=['GET'])
def history():
    try:
        history_path = os.path.join(app.static_folder, 'results', 'history.json')
        history_data = []
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history_data = json.load(f)
        return jsonify({'history': history_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/config', methods=['GET'])
def get_config():
    return jsonify({'port': config['server']['port']})

def find_available_port(start_port):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
            port += 1

if __name__ == '__main__':
    if not os.path.exists(os.path.join(app.static_folder, 'results')):
        os.makedirs(os.path.join(app.static_folder, 'results'))

    available_port = find_available_port(config['server']['port'])
    config['server']['port'] = available_port

    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)

    app.run(host=config['server']['host'], port=available_port, debug=True)
