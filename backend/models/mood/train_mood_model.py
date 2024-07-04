import os
import sys
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_tuner import Hyperband
import yaml
import logging

# Add the backend directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.data_utils import load_fer_data, save_cache

logging.basicConfig(level=logging.DEBUG)

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_dir = config['datasets']['fer_train']
val_data_dir = config['datasets']['fer_validation']
cache_dir = 'cache/FER'
model_dir = 'models/mood'
final_model_save_path = os.path.join(model_dir, 'final_trained_model_with_features.keras')

cache_only = False  # Change this to control whether to load only from cache or not

# Load or extract features
try:
    if not os.path.exists(os.path.join(cache_dir, 'features_train.npy')) or not os.path.exists(os.path.join(cache_dir, 'features_val.npy')):
        raise FileNotFoundError
    train_features = np.load(os.path.join(cache_dir, 'features_train.npy'))
    val_features = np.load(os.path.join(cache_dir, 'features_val.npy'))
    train_labels = np.load(os.path.join(cache_dir, 'labels_train.npy'))
    val_labels = np.load(os.path.join(cache_dir, 'labels_val.npy'))
    logging.info("Cache loaded successfully.")
except (FileNotFoundError, ValueError):
    logging.info("Cache not found or invalid. Extracting features from images.")
    train_features, train_labels = load_fer_data(data_dir, config['datasets']['predictor_path'], cache_dir, split='train', cache_only=cache_only)
    val_features, val_labels = load_fer_data(val_data_dir, config['datasets']['predictor_path'], cache_dir, split='val', cache_only=cache_only)
    if not cache_only:
        save_cache(train_features, val_features, train_labels, val_labels, cache_dir)
    logging.info("Features and labels extracted and saved to cache.")

# Convert labels to one-hot encoding
num_classes = config['model_params']['n_classes']
train_labels = to_categorical(train_labels, num_classes=num_classes)
val_labels = to_categorical(val_labels, num_classes=num_classes)

# Hyperparameter tuning function
def build_model(hp):
    input_shape_features = train_features.shape[1:]
    input_features = Input(shape=input_shape_features)
    x = Flatten()(input_features)
    for i in range(hp.Int('num_layers', 2, 5)):
        x = Dense(units=hp.Int(f'units_{i}', min_value=128, max_value=512, step=32))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=hp.Float(f'alpha_{i}', min_value=0.1, max_value=0.3))(x)
        x = Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5))(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_features, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = Hyperband(build_model, objective='val_accuracy', max_epochs=10, factor=3, directory=cache_dir, project_name='mood_tuning')

callbacks = [EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)]
tuner.search(train_features, train_labels, epochs=50, validation_data=(val_features, val_labels), callbacks=callbacks)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
logging.info(f"Best hyperparameters: {best_hp.values}")

# Build the final model with the best hyperparameters
model = build_model(best_hp)

# Define final callbacks
final_callbacks = [
    ModelCheckpoint(final_model_save_path, monitor='val_accuracy', save_best_only=True, mode='max'),
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
]

# Train the final model
model.fit(train_features, train_labels, epochs=100, batch_size=64, validation_data=(val_features, val_labels), callbacks=final_callbacks)
logging.info("Training completed.")

# Save the trained model architecture as an image
from tensorflow.keras.utils import plot_model

plot_model(model, to_file=os.path.join(cache_dir, 'student_model.png'), show_shapes=True, show_layer_names=True)
logging.info("Model architecture saved as 'student_model.png'.")
