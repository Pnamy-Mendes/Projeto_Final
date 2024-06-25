import os
import sys
import socket
import yaml
import base64
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from datetime import datetime
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.helpers import load_config, find_available_port, preprocess_image, draw_landmarks, extract_features, setup_tensorflow_gpu

app = Flask(__name__, static_url_path='/static')
CORS(app)

# Load configuration
config = load_config(os.path.join(os.path.dirname(__file__), '../../config.yaml'))

# Ensure model directory exists
model_dir = os.path.join(os.path.dirname(__file__), '../../models/mood/models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Load the trained teacher model
teacher_model_path = os.path.join(model_dir, 'teacher_model.keras')
if not os.path.exists(teacher_model_path):
    raise FileNotFoundError(f"Model file not found: {teacher_model_path}. Please ensure the file is in place.")

teacher_model = tf.keras.models.load_model(teacher_model_path)

mood_label_map = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_mood', methods=['POST'])
def predict_mood():
    try:
        data = request.json
        image_data = data['image']
        image, landmarks, error = preprocess_image(image_data)
        if error:
            print(f"Preprocessing error: {error}")
            return jsonify({'error': error})

        # Extract features from the image and landmarks
        features = extract_features(image, landmarks)
        features = np.expand_dims(features.astype('float32') / 255.0, axis=0)

        # Predict with the teacher model
        image_expanded = np.expand_dims(image.astype('float32') / 255.0, axis=0)
        prediction = teacher_model.predict(image_expanded)
        mood_idx = np.argmax(prediction[0])
        mood = mood_label_map[mood_idx]
        confidence = int(np.max(prediction[0]) * 100)

        print(f"Prediction: {prediction[0]}, Mood: {mood}, Confidence: {confidence}")

        # Draw landmarks on the image
        image_with_landmarks = draw_landmarks(image.copy(), landmarks)

        # Save the image with landmarks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_image_path = os.path.join('static/results', f"result_{timestamp}.png")
        cv2.imwrite(result_image_path, image_with_landmarks)

        # Save the prediction data
        history_path = os.path.join('static/results', 'history.json')
        history_data = []
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history_data = json.load(f)
        history_data.append({
            'timestamp': timestamp,
            'mood': mood,
            'confidence': confidence,
            'image_path': f"static/results/result_{timestamp}.png"
        })
        with open(history_path, 'w') as f:
            json.dump(history_data, f)

        return jsonify({
            'mood': mood,
            'confidence': confidence,
            'image_path': f"/static/results/result_{timestamp}.png"
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    try:
        history_path = os.path.join('static/results', 'history.json')
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

if __name__ == '__main__':
    if not os.path.exists('static/results'):
        os.makedirs('static/results')

    available_port = find_available_port(config['server']['port'])
    config['server']['port'] = available_port

    with open(os.path.join(os.path.dirname(__file__), '../../config.yaml'), 'w') as file:
        yaml.safe_dump(config, file)

    setup_tensorflow_gpu()

    app.run(host=config['server']['host'], port=available_port, debug=True, ssl_context=(
        os.path.join(os.path.dirname(__file__), 'cert.pem'), 
        os.path.join(os.path.dirname(__file__), 'key.pem')))
