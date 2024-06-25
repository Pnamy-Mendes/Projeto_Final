import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.helpers import load_config, setup_tensorflow_gpu, print_memory_usage
from utils.data_loader import load_data
from models.age_gender_race.multi_output_model import RobustCNN

# Load configuration
print("Loading configuration")
config = load_config('config.yaml')
print(f"Configuration loaded: {config}")

# Set up TensorFlow GPU configuration
print("Setting up TensorFlow GPU")
setup_tensorflow_gpu()

# Load datasets
print("Loading datasets")
try:
    data_dir = config['datasets']['age_gender_race_data']
    images, ages, genders, races, landmarks = load_data(data_dir)  # Assuming landmarks are also loaded here
    print(f"Loaded {len(images)} images")

    print("Splitting data into training and validation sets")
    X_train, X_val, landmarks_train, landmarks_val, age_train, age_val, gender_train, gender_val, race_train, race_val = train_test_split(
        images, landmarks, ages, genders, races, test_size=config['validation_split'], random_state=42
    )

    # One-hot encode the race labels
    race_encoder = OneHotEncoder(sparse=False)
    race_train = race_encoder.fit_transform(race_train.reshape(-1, 1))
    race_val = race_encoder.transform(race_val.reshape(-1, 1))

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Data loading failed. Check the data_loader function and dataset paths.")
    sys.exit(1)

# Initialize and compile model
print("Initializing the model")
model_path = "models/age_gender_race_model.keras"
robust_cnn = RobustCNN(model_path=model_path)

# Training callbacks
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

# Train model
print("Starting model training")
try:
    history = robust_cnn.model.fit(
        x={'image_input': X_train, 'landmark_input': landmarks_train},
        y={'age_output': age_train, 'gender_output': gender_train, 'race_output': race_train},
        validation_data=(
            {'image_input': X_val, 'landmark_input': landmarks_val},
            {'age_output': age_val, 'gender_output': gender_val, 'race_output': race_val}
        ),
        epochs=100,
        batch_size=32,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    print("Model training completed")
except Exception as e:
    print(f"Error during model training: {e}")
    print("Model training failed.")
    sys.exit(1)
