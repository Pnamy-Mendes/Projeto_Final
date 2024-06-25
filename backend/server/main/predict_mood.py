import os
import cv2
import numpy as np
import tensorflow as tf
from flask import jsonify
from datetime import datetime
from utils.helpers import preprocess_image, draw_landmarks, extract_features, load_config

# Load configuration
config = load_config('config.yaml')

# Load the trained teacher model
model_dir = "./models/mood/models"
teacher_model_path = os.path.join(model_dir, 'teacher_model.keras')
teacher_model = tf.keras.models.load_model(teacher_model_path)

mood_label_map = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}

def predict_mood(request):
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
