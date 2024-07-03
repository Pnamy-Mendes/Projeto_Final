import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, LeakyReLU, Input, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import keras_tuner as kt
import dlib
import cv2
import yaml

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define paths
data_dir = config['datasets']['fer_train']
predictor_path = config['datasets']['predictor_path']
cache_dir = 'models/mood'
teacher_model_path = os.path.join(cache_dir, 'teacher_model_with_features.keras')

# Function to calculate relative features
def calculate_relative_features(landmarks):
    if len(landmarks) == 0:
        return np.zeros(12)
    try:
        mouth_height = np.linalg.norm(landmarks[62] - landmarks[66]) / np.linalg.norm(landmarks[36] - landmarks[45])
        mouth_width = np.linalg.norm(landmarks[60] - landmarks[64]) / np.linalg.norm(landmarks[36] - landmarks[45])
        eye_distance = np.linalg.norm(landmarks[36] - landmarks[45]) / np.linalg.norm(landmarks[36] - landmarks[45])
        eyebrow_distance = np.linalg.norm(landmarks[19] - landmarks[24]) / np.linalg.norm(landmarks[36] - landmarks[45])
        nose_length = np.linalg.norm(landmarks[27] - landmarks[33]) / np.linalg.norm(landmarks[36] - landmarks[45])
        left_eye_ratio = np.linalg.norm(landmarks[37] - landmarks[41]) / np.linalg.norm(landmarks[36] - landmarks[39])
        right_eye_ratio = np.linalg.norm(landmarks[43] - landmarks[47]) / np.linalg.norm(landmarks[42] - landmarks[45])
        eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
        mouth_angle = np.degrees(np.arctan2(landmarks[54][1] - landmarks[48][1], landmarks[54][0] - landmarks[48][0]))
        inner_lip_distance = np.linalg.norm(landmarks[62] - landmarks[66]) / np.linalg.norm(landmarks[36] - landmarks[45])
        outer_lip_corners_up = 1 if landmarks[54][1] < landmarks[48][1] else 0
        outer_lip_corners_down = 1 if landmarks[54][1] > landmarks[48][1] else 0

        features = np.array([
            mouth_height, mouth_width, eye_distance, eyebrow_distance, nose_length, eye_ratio,
            left_eye_ratio, right_eye_ratio, mouth_angle, inner_lip_distance,
            outer_lip_corners_up, outer_lip_corners_down
        ])
        return features
    except Exception as e:
        print(f"Error calculating relative features: {e}")
        return np.zeros(12)

# Function to extract features from images
def extract_features(image_path, predictor_path):
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        
        if len(rects) > 0:
            shape = predictor(gray, rects[0])
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            relative_features = calculate_relative_features(landmarks)
            return image, relative_features
        else:
            return image, np.zeros(12)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros((48, 48, 3)), np.zeros(12)

# Function to build the VGG16 model with additional features and hyperparameter tuning
def build_teacher_model(hp):
    image_input = Input(shape=(48, 48, 3), name='image_input')
    features_input = Input(shape=(12,), name='features_input')
    
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=image_input)
    x = base_model.output
    x = Flatten()(x)
    
    combined = Concatenate()([x, features_input])
    x = Dense(hp.Int('units', min_value=256, max_value=512, step=64), activation='relu')(combined)
    x = Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1))(x)
    predictions = Dense(7, activation='softmax')(x)

    model = Model(inputs=[image_input, features_input], outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Hyperparameter Tuning
tuner = kt.Hyperband(
    build_teacher_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='hyperband',
    project_name='mood_teacher_tuning'
)

# Function to load data and extract features
def load_data(data_dir, predictor_path, batch_size=64):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=False
    )
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    train_features = []
    val_features = []

    for filename in train_generator.filenames:
        image_path = os.path.join(data_dir, 'training', filename)
        _, features = extract_features(image_path, predictor_path)
        train_features.append(features)

    for filename in validation_generator.filenames:
        image_path = os.path.join(data_dir, 'validation', filename)
        _, features = extract_features(image_path, predictor_path)
        val_features.append(features)

    return (train_generator, np.array(train_features)), (validation_generator, np.array(val_features))

# Load data and extract features
(train_generator, train_features), (validation_generator, val_features) = load_data(data_dir, predictor_path)

# Perform hyperparameter tuning
stop_early = EarlyStopping(monitor='val_loss', patience=5)
tuner.search([train_generator, train_features], epochs=10, validation_data=([validation_generator, val_features]), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the dense layer is {best_hps.get('units')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    [train_generator, train_features], 
    epochs=100, 
    validation_data=([validation_generator, val_features]), 
    callbacks=[stop_early]
)

# Save the trained model
model.save(teacher_model_path)
print(f"Teacher model saved at {teacher_model_path}")

# Save the model architecture as an image
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='teacher_model_with_features.png', show_shapes=True, show_layer_names=True)
print("Model architecture saved as 'teacher_model_with_features.png'.")
