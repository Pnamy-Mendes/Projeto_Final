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
import visualkeras

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define paths
train_data_dir = config['datasets']['fer_train']
val_data_dir = config['datasets']['fer_validation']
predictor_path = config['datasets']['predictor_path']
cache_dir = 'models/mood'
teacher_model_path = os.path.join(cache_dir, 'teacher_model_with_features.keras')

# Updated calculate_relative_features function
def calculate_relative_features(landmarks):
    if len(landmarks) == 0:
        return np.zeros(12)
    try:
        # Calculate distances relative to the face width (distance between eyes)
        face_width = np.linalg.norm(landmarks[36] - landmarks[45])

        mouth_height = np.linalg.norm(landmarks[62] - landmarks[66]) / face_width
        mouth_width = np.linalg.norm(landmarks[60] - landmarks[64]) / face_width
        eye_distance = np.linalg.norm(landmarks[36] - landmarks[45]) / face_width
        eyebrow_distance = np.linalg.norm(landmarks[19] - landmarks[24]) / face_width
        nose_length = np.linalg.norm(landmarks[27] - landmarks[33]) / face_width
        left_eye_ratio = np.linalg.norm(landmarks[37] - landmarks[41]) / np.linalg.norm(landmarks[36] - landmarks[39])
        right_eye_ratio = np.linalg.norm(landmarks[43] - landmarks[47]) / np.linalg.norm(landmarks[42] - landmarks[45])
        eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
        mouth_angle = np.degrees(np.arctan2(landmarks[54][1] - landmarks[48][1], landmarks[54][0] - landmarks[48][0]))
        inner_lip_distance = np.linalg.norm(landmarks[62] - landmarks[66]) / face_width
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
        print(f"Processing image: {image_path}")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        
        rects = detector(image, 1)
        
        if len(rects) > 0:
            shape = predictor(image, rects[0])
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            relative_features = calculate_relative_features(landmarks)
            
            # Convert grayscale image to RGB by repeating the single channel
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Resize the image to (48, 48, 3)
            resized_image = cv2.resize(image_rgb, (48, 48))
            
            return resized_image, relative_features
        else:
            print(f"No faces detected in image: {image_path}")
            return np.zeros((48, 48, 3)), np.zeros(12)
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return np.zeros((48, 48, 3)), np.zeros(12)

# Function to build the VGG16 model with additional features and hyperparameter tuning
def build_teacher_model(hp):
    image_input = Input(shape=(48, 48, 3), name='image_input')  # Updated shape to handle RGB images
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

# Function to load data and extract features with a max_images_per_class parameter
def load_data(data_dir, predictor_path, subset, batch_size=64, max_images_per_class=None):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(48, 48),
        color_mode='grayscale',  # Ensure grayscale mode
        batch_size=batch_size,
        class_mode='categorical',
        subset=subset,
        shuffle=False
    )

    images = []
    labels = []
    features = []
    class_counts = {}

    print(f"Processing {subset} data...")
    for filename in generator.filenames:
        class_label = filename.split('/')[0]
        if max_images_per_class is not None:
            if class_label not in class_counts:
                class_counts[class_label] = 0
            if class_counts[class_label] >= max_images_per_class:
                continue
            class_counts[class_label] += 1

        image_path = os.path.join(data_dir, filename)
        print(f"{subset} image path: {image_path}")
        image, feature = extract_features(image_path, predictor_path)

        if image is not None and feature is not None and feature.shape == (12,):  # Ensure valid features are returned
            images.append(image)
            features.append(feature)

            label = generator.class_indices[class_label]
            labels.append(label)
        else:
            print(f"Skipping invalid data for image: {image_path}")
            if feature is not None:
                print(f"Invalid feature shape: {feature.shape} for image: {image_path}")

    # Print detailed information about images and features
    print("Images and Features shapes before conversion to NumPy array:")
    for idx, img in enumerate(images):
        print(f"Image {idx} shape: {img.shape}")
    for idx, feat in enumerate(features):
        print(f"Feature {idx} shape: {feat.shape}")

    # Ensure all features have the correct shape before converting to numpy arrays
    valid_features = [f for f in features if f.shape == (12,)]
    invalid_features_count = len(features) - len(valid_features)
    print(f"Number of valid features: {len(valid_features)}")
    print(f"Number of invalid features: {invalid_features_count}")

    print(f"Converting {len(images)} images and {len(valid_features)} features to NumPy arrays...")
    return np.array(images), np.array(valid_features), to_categorical(labels, num_classes=7)

# Load data and extract features
train_images, train_features, train_labels = load_data(train_data_dir, predictor_path, 'training', max_images_per_class=None)
val_images, val_features, val_labels = load_data(val_data_dir, predictor_path, 'validation', max_images_per_class=None)

# Perform hyperparameter tuning
stop_early = EarlyStopping(monitor='val_loss', patience=5)
tuner.search([train_images, train_features], train_labels, epochs=10, validation_data=([val_images, val_features], val_labels), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the dense layer is {best_hps.get('units')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    [train_images, train_features], 
    train_labels, 
    epochs=100, 
    validation_data=([val_images, val_features], val_labels), 
    callbacks=[stop_early]
)

# Save the trained model
model.save(teacher_model_path)
print(f"Teacher model saved at {teacher_model_path}")

# Save the model architecture as an image
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='teacher_model_with_features.png', show_shapes=True, show_layer_names=True)
print("Model architecture saved as 'mood_teacher_model_with_features.png'.")

# Visualize the model architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
print("Model architecture plot saved as 'mood_model_architecture.png'")

# Visualize using visualkeras
visualkeras.layered_view(model, to_file='model_visualization.png').show()
print("Model visualization saved as 'mood_model_visualization.png'")