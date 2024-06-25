import os
import yaml
import cv2
import numpy as np
import dlib
import base64
import socket
import psutil  # For checking system memory
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_config(config_path):
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(f"Configuration loaded: {config}")
    return config

def find_available_port(start_port):
    print(f"Finding available port starting from {start_port}")
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                print(f"Found available port: {port}")
                return port
            port += 1

def setup_tensorflow_gpu():
    print("Setting up TensorFlow GPU")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found. Training will be done on CPU.")

def preprocess_image(image_data):
    config = load_config('config.yaml')
    detector = dlib.get_frontal_face_detector()
    predictor_path = config['datasets']['predictor_path']
    predictor = dlib.shape_predictor(predictor_path)
    
    try:
        print("Preprocessing image")
        image = base64.b64decode(image_data.split(",")[1])
        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if len(rects) == 0:
            print("No face detected")
            return None, None, "No face detected"
        shape = predictor(gray, rects[0])
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        image_resized = cv2.resize(image, (48, 48))
        return image_resized, landmarks, None
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None, None, f"Preprocessing error: {e}"

def draw_landmarks(image, landmarks):
    print("Drawing landmarks")
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    return image

def print_memory_usage():
    mem = psutil.virtual_memory()
    print(f"Total memory: {mem.total / (1024 ** 3):.2f} GB")
    print(f"Available memory: {mem.available / (1024 ** 3):.2f} GB")
    print(f"Used memory: {mem.used / (1024 ** 3):.2f} GB")
    print(f"Memory usage: {mem.percent}%")

def data_loader(data_path, validation_split, input_shape):
    print(f"Loading data from {data_path}")
    try:
        images = []
        ages = []
        genders = []
        races = []

        for filename in os.listdir(data_path):
            if filename.endswith('.jpg'):
                parts = filename.split('_')
                if len(parts) >= 4:
                    age = int(parts[0])
                    gender = int(parts[1])
                    race = int(parts[2])
                    
                    img_path = os.path.join(data_path, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Could not read image {img_path}")
                        continue
                    
                    img = cv2.resize(img, (input_shape[0], input_shape[1]))
                    images.append(img)
                    ages.append(age)
                    genders.append(gender)
                    races.append(race)

                    if len(images) % 1000 == 0:
                        print(f"Processed {len(images)} images")
                        print_memory_usage()

        images = np.array(images, dtype='float32') / 255.0
        ages = np.array(ages)
        genders = np.array(genders)
        races = tf.keras.utils.to_categorical(races, num_classes=len(np.unique(races)))

        print("Splitting data into training and validation sets")
        x_train, x_val, y_train_age, y_val_age, y_train_gender, y_val_gender, y_train_race, y_val_race = train_test_split(
            images, ages, genders, races, test_size=validation_split, random_state=42)

        train_data = (x_train, {'age_output': y_train_age, 'gender_output': y_train_gender, 'race_output': y_train_race})
        val_data = (x_val, {'age_output': y_val_age, 'gender_output': y_val_gender, 'race_output': y_val_race})

        print(f"Data loaded successfully: {len(images)} images")
        print_memory_usage()
        return train_data, val_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
