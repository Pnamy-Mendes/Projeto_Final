import os
import cv2
import numpy as np
import tensorflow as tf
from flask import jsonify
from datetime import datetime
from utils.helpers import preprocess_image, draw_landmarks, load_config

config = load_config('config.yaml')

model_dir = "./models/age_gender_race/models"
multi_output_model_path = os.path.join(model_dir, 'final_trained_model.keras')
multi_output_model = tf.keras.models.load_model(multi_output_model_path)

age_model_path = 'models/age_model_final_save.keras'
gender_model_path = 'models/gender_model.keras'
race_model_path = 'models/race_model_final_saveA.keras'

age_model = tf.keras.models.load_model(age_model_path)
gender_model = tf.keras.models.load_model(gender_model_path)
race_model = tf.keras.models.load_model(race_model_path)

race_label_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}

def predict_age_gender_race(request):
    try:
        data = request.json
        image_data = data['image']
        image, landmarks, error = preprocess_image(image_data)
        if error:
            print(f"Preprocessing error: {error}")
            return jsonify({'error': error})

        image_expanded = np.expand_dims(image.astype('float32') / 255.0, axis=0)
        landmarks_expanded = np.expand_dims(landmarks, axis=0)

        mood_pred, age_pred, gender_pred, race_pred = multi_output_model.predict([image_expanded, landmarks_expanded])
        age = int(age_pred[0][0])
        gender = 'male' if gender_pred[0][0] > 0.5 else 'female'
        race_idx = np.argmax(race_pred[0])
        race = race_label_map.get(race_idx, "Unknown")

        print(f"Prediction: Age: {age}, Gender: {gender}, Race: {race}")

        image_with_landmarks = draw_landmarks(image.copy(), landmarks)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_image_path = os.path.join('static/results', f"result_{timestamp}.png")
        cv2.imwrite(result_image_path, image_with_landmarks)

        return jsonify({
            'age': age,
            'gender': gender,
            'race': race,
            'image_path': f"/static/results/result_{timestamp}.png"
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500
