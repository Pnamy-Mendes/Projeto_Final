import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Concatenate, Dense, Flatten, Input, Dropout, BatchNormalization, 
    GlobalAveragePooling2D, Conv2D, MaxPooling2D
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
    train_gen, val_gen, steps_per_epoch_train, steps_per_epoch_val = data_loader(
        data_dir, validation_split, input_shape, config, batch_size, max_images=max_images, use_cache_only=use_cache_only
    )
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

def build_model(hp, output_type=None):
    input_shape_images = (256, 256, 3)
    input_shape_landmarks = (68, 2)
    input_shape_features = (13,)  # Update input shape to match actual data shape
    input_shape_additional = (6,)  # Adjust according to the number of additional features
    num_races = 5

    image_input = Input(shape=input_shape_images, name='image_input')
    landmark_input = Input(shape=input_shape_landmarks, name='landmark_input')
    features_input = Input(shape=input_shape_features, name='features_input')
    additional_input = Input(shape=input_shape_additional, name='additional_input')

    # Enhanced model with more convolutional layers
    x = Conv2D(96, (7, 7), strides=(4, 4), padding='same', activation='relu')(image_input)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Conv2D(256, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = GlobalAveragePooling2D()(x)

    flat_landmarks = Flatten()(landmark_input)
    concatenated_features = Concatenate(name='concatenate_features')([x, flat_landmarks, features_input, additional_input])

    dense_concat = Dense(hp.Int('dense_units', min_value=128, max_value=512, step=64), activation='relu', name='dense_concat')(concatenated_features)
    dropout = Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1), name='dropout_concat')(dense_concat)
    batch_norm = BatchNormalization(name='batch_norm_concat')(dropout)

    if output_type == 'age':
        final_output = Dense(1, name='final_output_age', dtype=tf.float32)(batch_norm)
        loss = 'mean_squared_error'
        metrics = ['mae']
    elif output_type == 'gender':
        final_output = Dense(1, activation='sigmoid', name='final_output_gender', dtype=tf.float32)(batch_norm)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    elif output_type == 'race':
        final_output = Dense(num_races, activation='softmax', name='final_output_race', dtype=tf.float32)(batch_norm)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:
        raise ValueError("output_type must be 'age', 'gender', or 'race'")

    model = Model(inputs=[image_input, landmark_input, features_input, additional_input], outputs=final_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])),
        loss=loss,
        metrics=metrics
    )
    return model

# Hyperparameter Tuning for Age Output
print("Starting hyperparameter tuning for age output")
stop_early_age = EarlyStopping(monitor='val_mae', patience=5, mode='min')
tuner_age = kt.Hyperband(
    lambda hp: build_model(hp, 'age'),
    objective='val_mae',
    max_epochs=20,
    factor=3,
    directory='hyperband_teacher',
    project_name='age_tuning'
)
tuner_age.search(train_gen, epochs=20, steps_per_epoch=steps_per_epoch_train, validation_data=val_gen, validation_steps=steps_per_epoch_val, callbacks=[stop_early_age])
best_hps_age = tuner_age.get_best_hyperparameters(num_trials=1)[0]
print(f"Hyperparameter tuning for age completed.")

# Hyperparameter Tuning for Gender Output
print("Starting hyperparameter tuning for gender output")
stop_early_gender = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
tuner_gender = kt.Hyperband(
    lambda hp: build_model(hp, 'gender'),
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='hyperband_teacher',
    project_name='gender_tuning'
)
tuner_gender.search(train_gen, epochs=20, steps_per_epoch=steps_per_epoch_train, validation_data=val_gen, validation_steps=steps_per_epoch_val, callbacks=[stop_early_gender])
best_hps_gender = tuner_gender.get_best_hyperparameters(num_trials=1)[0]
print(f"Hyperparameter tuning for gender completed.")

# Hyperparameter Tuning for Race Output
print("Starting hyperparameter tuning for race output")
stop_early_race = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
tuner_race = kt.Hyperband(
    lambda hp: build_model(hp, 'race'),
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='hyperband_teacher',
    project_name='race_tuning'
)

# One-hot encode the race labels for the hyperparameter tuning process
y_train_race_onehot = tf.keras.utils.to_categorical(y_train_race, num_classes=5)
y_val_race_onehot = tf.keras.utils.to_categorical(y_val_race, num_classes=5)

tuner_race.search(train_gen, epochs=20, steps_per_epoch=steps_per_epoch_train, validation_data=val_gen, validation_steps=steps_per_epoch_val, callbacks=[stop_early_race])
best_hps_race = tuner_race.get_best_hyperparameters(num_trials=1)[0]
print(f"Hyperparameter tuning for race completed.")

# Combine the best hyperparameters
def build_age_gender_race_model(input_shape_images, input_shape_landmarks, input_shape_features, input_shape_additional, num_races):
    try:
        image_input = Input(shape=input_shape_images, name='image_input')
        landmark_input = Input(shape=input_shape_landmarks, name='landmark_input')
        features_input = Input(shape=input_shape_features, name='features_input')
        additional_input = Input(shape=input_shape_additional, name='additional_input')

        # Enhanced model with more convolutional layers
        x = Conv2D(96, (7, 7), strides=(4, 4), padding='same', activation='relu')(image_input)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = Conv2D(256, (5, 5), padding='same', activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = GlobalAveragePooling2D()(x)

        flat_landmarks = Flatten()(landmark_input)
        concatenated_features = Concatenate(name='concatenate_features')([x, flat_landmarks, features_input, additional_input])

        dense_concat = Dense(best_hps_age.get('dense_units'), activation='relu', name='dense_concat')(concatenated_features)
        dropout = Dropout(best_hps_age.get('dropout'), name='dropout_concat')(dense_concat)
        batch_norm = BatchNormalization(name='batch_norm_concat')(dropout)

        final_output_age = Dense(1, name='final_output_age', dtype=tf.float32)(batch_norm)
        final_output_gender = Dense(1, activation='sigmoid', name='final_output_gender', dtype=tf.float32)(batch_norm)
        final_output_race = Dense(num_races, activation='softmax', name='final_output_race', dtype=tf.float32)(batch_norm)

        model = Model(inputs=[image_input, landmark_input, features_input, additional_input], outputs=[final_output_age, final_output_gender, final_output_race])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps_age.get('learning_rate')),
            loss={
                'final_output_age': 'mean_squared_error',
                'final_output_gender': 'binary_crossentropy',
                'final_output_race': 'categorical_crossentropy'
            },
            metrics={
                'final_output_age': 'mae',
                'final_output_gender': 'accuracy',
                'final_output_race': 'accuracy'
            }
        )
        return model
    except Exception as e:
        print(f"Error building the model: {e}")
        raise

model = build_age_gender_race_model(
    input_shape_images=(256, 256, 3),
    input_shape_landmarks=(68, 2),
    input_shape_features=(13,),  # Updated to match the actual input data shape
    input_shape_additional=(6,),  # Adjust according to the number of additional features
    num_races=5
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
        train_gen, 
        epochs=100,
        steps_per_epoch=steps_per_epoch_train,
        validation_data=val_gen,
        validation_steps=steps_per_epoch_val,
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
