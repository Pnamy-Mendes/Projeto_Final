import sys
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense, Flatten, Input, Dropout, BatchNormalization, 
    GlobalAveragePooling2D, Conv2D, MaxPooling2D, Reshape, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.mixed_precision import set_global_policy
import yaml
import keras_tuner as kt

# Adjust the backend directory import
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(backend_dir)

try:
    from utils.age_gender_race_helpers import load_config, setup_tensorflow_gpu, data_loader
    from utils.feature_extraction import extract_features
    print("Successfully imported modules.")
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    print("Make sure the utils directory contains the necessary modules.")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    exit(1)

# Argument parser for hyperparameter tuning
parser = argparse.ArgumentParser(description="Train and tune the age-gender-race model.")
parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
args = parser.parse_args()

# Load configuration
config = load_config(os.path.join(backend_dir, 'config.yaml'))

# Enable mixed precision training
set_global_policy('mixed_float16')

# Load data
print("Loading data...")
data_dir = config['datasets']['age_gender_race_data']
validation_split = config['validation_split']
input_shape = config['model_params']['input_shape']
batch_size = config['model_params']['batch_size']
max_images = 5000  # Limit the number of images to avoid memory issues
use_cache_only = True  # Load only cached images

try:
    x_train, x_val, y_train, y_val = data_loader(
        data_dir, validation_split, input_shape, config, batch_size, max_images=max_images, use_cache_only=use_cache_only
    )
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

def create_dataset(x, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=len(x)).batch(batch_size)
    return dataset

train_dataset = create_dataset(x_train, y_train, batch_size)
val_dataset = create_dataset(x_val, y_val, batch_size)

def build_model(hp):
    input_shape_combined = (x_train.shape[1],)
    combined_input = Input(shape=input_shape_combined, name='combined_input', dtype='float32')

    # Extract the image part from the combined input
    img_flat_size = 256 * 256 * 3

    x = Reshape((256, 256, 3))(combined_input[:, :img_flat_size])
    features = combined_input[:, img_flat_size:]

    # Model layers
    x = Conv2D(96, (7, 7), strides=(4, 4), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Conv2D(256, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = GlobalAveragePooling2D()(x)

    concatenated_features = Concatenate(name='concatenate_features')([x, features])

    dense_concat = Dense(hp.Int('dense_units', min_value=128, max_value=512, step=64), activation='relu', name='dense_concat')(concatenated_features)
    dropout = Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1), name='dropout_concat')(dense_concat)
    batch_norm = BatchNormalization(name='batch_norm_concat')(dropout)

    final_output = Dense(3, name='final_output', dtype=tf.float32)(batch_norm)

    model = Model(inputs=combined_input, outputs=final_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])),
        loss='mse',  # Simplified combined loss
        metrics=['mae', 'accuracy']
    )
    return model

if args.tune:
    # Hyperparameter Tuning
    print("Starting hyperparameter tuning")
    stop_early = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=20,
        factor=3,
        directory='hyperband_teacher',
        project_name='combined_tuning'
    )
    tuner.search(
        train_dataset, 
        epochs=20, 
        validation_data=val_dataset, 
        callbacks=[stop_early]
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Hyperparameter tuning completed.")
else:
    # Load pre-tuned hyperparameters
    with open('best_hyperparameters.yaml', 'r') as file:
        best_hps = yaml.safe_load(file)

def build_combined_model(input_shape_combined, best_hps):
    try:
        combined_input = Input(shape=input_shape_combined, name='combined_input', dtype='float32')

        # Extract the image part from the combined input
        img_flat_size = 256 * 256 * 3

        x = Reshape((256, 256, 3))(combined_input[:, :img_flat_size])
        features = combined_input[:, img_flat_size:]

        # Enhanced model with more convolutional layers
        x = Conv2D(96, (7, 7), strides=(4, 4), padding='same', activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = Conv2D(256, (5, 5), padding='same', activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = GlobalAveragePooling2D()(x)

        concatenated_features = Concatenate(name='concatenate_features')([x, features])

        dense_concat = Dense(best_hps.get('dense_units'), activation='relu', name='dense_concat')(concatenated_features)
        dropout = Dropout(best_hps.get('dropout'), name='dropout_concat')(dense_concat)
        batch_norm = BatchNormalization(name='batch_norm_concat')(dropout)

        final_output = Dense(3, name='final_output', dtype=tf.float32)(batch_norm)

        model = Model(inputs=combined_input, outputs=final_output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate')),
            loss='mse',  # Simplified combined loss
            metrics=['mae', 'accuracy']
        )
        return model
    except Exception as e:
        print(f"Error building the model: {e}")
        raise

model = build_combined_model(
    input_shape_combined=(x_train.shape[1],),
    best_hps=best_hps
)

# Load existing model if exists
model_path = os.path.join(backend_dir, 'models', 'age_gender_race', 'models', 'teacher_model_best.keras')
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Loaded existing model for continued training.")

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
checkpoint_dir = os.path.join(backend_dir, 'models', 'age_gender_race', 'models')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, 'teacher_model_best.keras'), monitor='val_loss', save_best_only=True)

lr_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)

print("Starting model training")
try:
    history = model.fit(
        train_dataset, 
        epochs=100,
        validation_data=val_dataset,
        callbacks=[early_stopping, reduce_lr, checkpoint, lr_scheduler]
    )
except Exception as e:
    print(f"An error occurred during training: {e}")
    print("Exiting.")
    exit(1)

print("Model training completed successfully")

# Save the final model
model.save(os.path.join(checkpoint_dir, 'teacher_model_final.keras'))
print("Teacher model training completed and saved.")
