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
    (x_train_img, x_train_features, y_train_age, y_train_gender, y_train_race), (x_val_img, x_val_features, y_val_age, y_val_gender, y_val_race) = data_loader(
        data_dir, validation_split, input_shape, config, batch_size, max_images=max_images, use_cache_only=use_cache_only
    )
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

def create_dataset(x_img, x_features, y_age, y_gender, y_race, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(({'image_input': x_img, 'feature_input': x_features}, {'age_output': y_age, 'gender_output': y_gender, 'race_output': y_race}))
    dataset = dataset.shuffle(buffer_size=len(x_img)).batch(batch_size)
    return dataset

train_dataset = create_dataset(x_train_img, x_train_features, y_train_age, y_train_gender, y_train_race, batch_size)
val_dataset = create_dataset(x_val_img, x_val_features, y_val_age, y_val_gender, y_val_race, batch_size)

def build_model(hp, output_units, name):
    img_input_shape = x_train_img.shape[1:]
    feature_input_shape = x_train_features.shape[1:]

    image_input = Input(shape=img_input_shape, name='image_input')
    feature_input = Input(shape=feature_input_shape, name='feature_input')

    # Simplified model layers
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(image_input)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = GlobalAveragePooling2D()(x)

    concatenated_features = Concatenate(name='concatenate_features')([x, feature_input])

    dense_concat = Dense(hp.Int('dense_units', min_value=64, max_value=256, step=64), activation='relu', name='dense_concat')(concatenated_features)
    dropout = Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1), name='dropout_concat')(dense_concat)
    batch_norm = BatchNormalization(name='batch_norm_concat')(dropout)

    if name == 'age':
        final_output = Dense(output_units, name=f'final_output_{name}', dtype=tf.float32)(batch_norm)
        loss = 'mse'
        metrics = ['mae']
    else:
        final_output = Dense(output_units, activation='softmax', name=f'final_output_{name}', dtype=tf.float32)(batch_norm)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']

    model = Model(inputs=[image_input, feature_input], outputs=final_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])),
        loss=loss,
        metrics=metrics
    )
    return model

def tune_and_train_model(name, output_units, y_train, y_val, epochs=10, max_trials=5):
    print(f"Starting hyperparameter tuning for {name}")
    tuner = kt.Hyperband(
        lambda hp: build_model(hp, output_units, name),
        objective='val_loss',
        max_epochs=epochs,
        factor=3,
        directory='hyperband_teacher',
        project_name=f'{name}_tuning'
    )
    stop_early = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    tuner.search(
        train_dataset.map(lambda x, y: (x, y[name + '_output'])), 
        epochs=epochs, 
        validation_data=val_dataset.map(lambda x, y: (x, y[name + '_output'])), 
        callbacks=[stop_early]
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Hyperparameter tuning completed for {name}.")

    model = build_model(best_hps, output_units, name)
    model.fit(
        train_dataset.map(lambda x, y: (x, y[name + '_output'])), 
        epochs=30, 
        validation_data=val_dataset.map(lambda x, y: (x, y[name + '_output'])), 
        callbacks=[stop_early, ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)]
    )
    return model, best_hps

# Tuning for age, gender, and race separately
age_model, age_hps = tune_and_train_model('age', 1, y_train_age, y_val_age)
gender_model, gender_hps = tune_and_train_model('gender', 2, y_train_gender, y_val_gender)
race_model, race_hps = tune_and_train_model('race', 5, y_train_race, y_val_race)

# Combined model building with best hyperparameters
def build_combined_model(img_input_shape, feature_input_shape, age_hps, gender_hps, race_hps):
    image_input = Input(shape=img_input_shape, name='image_input')
    feature_input = Input(shape=feature_input_shape, name='feature_input')

    # Shared convolutional base
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(image_input)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = GlobalAveragePooling2D()(x)

    concatenated_features = Concatenate(name='concatenate_features')([x, feature_input])

    # Age branch
    age_dense = Dense(age_hps.get('dense_units'), activation='relu', name='age_dense')(concatenated_features)
    age_dropout = Dropout(age_hps.get('dropout'), name='age_dropout')(age_dense)
    age_batch_norm = BatchNormalization(name='age_batch_norm')(age_dropout)
    age_output = Dense(1, name='age_output', dtype=tf.float32)(age_batch_norm)

    # Gender branch
    gender_dense = Dense(gender_hps.get('dense_units'), activation='relu', name='gender_dense')(concatenated_features)
    gender_dropout = Dropout(gender_hps.get('dropout'), name='gender_dropout')(gender_dense)
    gender_batch_norm = BatchNormalization(name='gender_batch_norm')(gender_dropout)
    gender_output = Dense(2, activation='softmax', name='gender_output', dtype=tf.float32)(gender_batch_norm)

    # Race branch
    race_dense = Dense(race_hps.get('dense_units'), activation='relu', name='race_dense')(concatenated_features)
    race_dropout = Dropout(race_hps.get('dropout'), name='race_dropout')(race_dense)
    race_batch_norm = BatchNormalization(name='race_batch_norm')(race_dropout)
    race_output = Dense(5, activation='softmax', name='race_output', dtype=tf.float32)(race_batch_norm)

    model = Model(inputs=[image_input, feature_input], outputs=[age_output, gender_output, race_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=['mse', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'],
        metrics=['mae', 'accuracy', 'accuracy']
    )
    return model

combined_model = build_combined_model(
    img_input_shape=x_train_img.shape[1:],
    feature_input_shape=x_train_features.shape[1:],
    age_hps=age_hps,
    gender_hps=gender_hps,
    race_hps=race_hps
)

# Load existing model if exists
model_path = os.path.join(backend_dir, 'models', 'age_gender_race', 'models', 'teacher_model_best.keras')
if os.path.exists(model_path):
    combined_model = load_model(model_path)
    print("Loaded existing model for continued training.")

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
checkpoint_dir = os.path.join(backend_dir, 'models', 'age_gender_race', 'models')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, 'teacher_model_best.keras'), monitor='val_loss', save_best_only=True)

lr_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)

print("Starting combined model training")
try:
    history = combined_model.fit(
        train_dataset, 
        epochs=100,
        validation_data=val_dataset,
        callbacks=[early_stopping, reduce_lr, checkpoint, lr_scheduler]
    )
except Exception as e:
    print(f"An error occurred during training: {e}")
    print("Exiting.")
    exit(1)

print("Combined model training completed successfully")

# Save the final model
combined_model.save(os.path.join(checkpoint_dir, 'teacher_model_final.keras'))
print("Teacher model training completed and saved.")
