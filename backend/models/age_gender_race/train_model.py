import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Concatenate, Dense, Flatten, Input, Dropout, BatchNormalization, 
                                     GlobalAveragePooling2D, Conv2D, MaxPooling2D, LeakyReLU)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.mixed_precision import global_policy, set_global_policy
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import time
import keras_tuner as kt

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(current_dir, '../..'))
    sys.path.append(backend_dir)
    print(f"Added {backend_dir} to Python path.")
except Exception as e:
    print(f"Error adding backend directory to Python path: {e}")
    exit(1)

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

try:
    config = load_config('config.yaml')
    print("Configuration loaded successfully.")
except Exception as e:
    print(f"Error loading configuration: {e}")
    exit(1)

try:
    setup_tensorflow_gpu()
    print("TensorFlow GPU setup successfully.")
except Exception as e:
    print(f"Error setting up TensorFlow GPU: {e}")
    exit(1)

# Enable mixed precision training
set_global_policy('mixed_float16')

try:
    data_dir = config['datasets']['age_gender_race_data']
    validation_split = config['validation_split']
    input_shape = config['model_params']['input_shape']
    feature_length = 8
    max_images = 1000
    use_cache_only = True  # Set to True to use cache only, set to False to load and process new images

    train_data, val_data = data_loader(data_dir, validation_split, input_shape, config, max_images=max_images, use_cache_only=use_cache_only)

    if train_data is None or val_data is None:
        print("Data loading failed. Exiting.")
        exit(1)

    x_train, y_train = train_data
    x_val, y_val = val_data

    if len(y_train) == 3:
        y_train_age, y_train_gender, y_train_race = y_train
        y_val_age, y_val_gender, y_val_race = y_val
    else:
        print("Unexpected structure of y_train and y_val")
        exit(1)

    print(f"Training data shapes - Images: {x_train[0].shape}, Landmarks: {x_train[1].shape}, Features: {x_train[2].shape}")
    print(f"Validation data shapes - Images: {x_val[0].shape}, Landmarks: {x_val[1].shape}, Features: {x_val[2].shape}")
    print(f"Training labels shapes - Age: {y_train_age.shape}, Gender: {y_train_gender.shape}, Race: {y_train_race.shape}")
    print(f"Validation labels shapes - Age: {y_val_age.shape}, Gender: {y_val_gender.shape}, Race: {y_val_race.shape}")
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

def build_model(hp, output_type=None):
    input_shape_images = (256, 256, 3)
    input_shape_landmarks = (68, 2)
    input_shape_features = (8,)
    num_races = 5

    image_input = Input(shape=input_shape_images, name='image_input')
    landmark_input = Input(shape=input_shape_landmarks, name='landmark_input')
    features_input = Input(shape=input_shape_features, name='features_input')

    x = Conv2D(hp.Int('conv_1_filter', min_value=32, max_value=96, step=16), (3, 3), padding='same')(image_input)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Conv2D(hp.Int('conv_2_filter', min_value=32, max_value=96, step=16), (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(hp.Int('conv_3_filter', min_value=32, max_value=128, step=16), (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = GlobalAveragePooling2D()(x)

    flat_landmarks = Flatten()(landmark_input)
    concatenated_features = Concatenate(name='concatenate_features')([x, flat_landmarks, features_input])

    dense_concat = Dense(hp.Int('dense_units', min_value=128, max_value=256, step=32), activation='relu', name='dense_concat')(concatenated_features)
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

    model = Model(inputs=[image_input, landmark_input, features_input], outputs=final_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
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
    max_epochs=10,
    factor=3,
    directory='hyperband',
    project_name='age_tuning'
)
tuner_age.search([x_train[0], x_train[1], x_train[2]], y_train_age, epochs=10, validation_data=([x_val[0], x_val[1], x_val[2]], y_val_age), callbacks=[stop_early_age])
best_hps_age = tuner_age.get_best_hyperparameters(num_trials=1)[0]
print(f"Hyperparameter tuning for age completed.")

# Hyperparameter Tuning for Gender Output
print("Starting hyperparameter tuning for gender output")
stop_early_gender = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
tuner_gender = kt.Hyperband(
    lambda hp: build_model(hp, 'gender'),
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='hyperband',
    project_name='gender_tuning'
)
tuner_gender.search([x_train[0], x_train[1], x_train[2]], y_train_gender, epochs=10, validation_data=([x_val[0], x_val[1], x_val[2]], y_val_gender), callbacks=[stop_early_gender])
best_hps_gender = tuner_gender.get_best_hyperparameters(num_trials=1)[0]
print(f"Hyperparameter tuning for gender completed.")

# Hyperparameter Tuning for Race Output
print("Starting hyperparameter tuning for race output")
stop_early_race = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
tuner_race = kt.Hyperband(
    lambda hp: build_model(hp, 'race'),
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='hyperband',
    project_name='race_tuning'
)

# One-hot encode the race labels for the hyperparameter tuning process
y_train_race_onehot = tf.keras.utils.to_categorical(y_train_race, num_classes=5)
y_val_race_onehot = tf.keras.utils.to_categorical(y_val_race, num_classes=5)

tuner_race.search([x_train[0], x_train[1], x_train[2]], y_train_race_onehot, epochs=10, validation_data=([x_val[0], x_val[1], x_val[2]], y_val_race_onehot), callbacks=[stop_early_race])
best_hps_race = tuner_race.get_best_hyperparameters(num_trials=1)[0]
print(f"Hyperparameter tuning for race completed.")

# Combine the best hyperparameters
def build_age_gender_race_model(input_shape_images, input_shape_landmarks, input_shape_features, num_races):
    try:
        image_input = Input(shape=input_shape_images, name='image_input')
        landmark_input = Input(shape=input_shape_landmarks, name='landmark_input')
        features_input = Input(shape=input_shape_features, name='features_input')

        x = Conv2D(best_hps_age.get('conv_1_filter'), (3, 3), padding='same')(image_input)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = Conv2D(best_hps_age.get('conv_2_filter'), (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(best_hps_age.get('conv_3_filter'), (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = GlobalAveragePooling2D()(x)

        flat_landmarks = Flatten()(landmark_input)
        concatenated_features = Concatenate(name='concatenate_features')([x, flat_landmarks, features_input])

        dense_concat = Dense(best_hps_age.get('dense_units'), activation='relu', name='dense_concat')(concatenated_features)
        dropout = Dropout(best_hps_age.get('dropout'), name='dropout_concat')(dense_concat)
        batch_norm = BatchNormalization(name='batch_norm_concat')(dropout)

        final_output_age = Dense(1, name='final_output_age', dtype=tf.float32)(batch_norm)
        final_output_gender = Dense(1, activation='sigmoid', name='final_output_gender', dtype=tf.float32)(batch_norm)
        final_output_race = Dense(num_races, activation='softmax', name='final_output_race', dtype=tf.float32)(batch_norm)

        model = Model(inputs=[image_input, landmark_input, features_input], outputs=[final_output_age, final_output_gender, final_output_race])
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
    input_shape_features=(8,),
    num_races=5
)

y_train_race_onehot = tf.keras.utils.to_categorical(y_train_race, num_classes=5)
y_val_race_onehot = tf.keras.utils.to_categorical(y_val_race, num_classes=5)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
lr_scheduler = LearningRateScheduler(lambda epoch: 1e-4 * 0.95 ** epoch)
checkpoint_dir = os.path.join('backend', 'models', 'age_gender_race', 'models')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, 'age_gender_race_model_best.keras'), monitor='val_loss', save_best_only=True)

epochs = 100
batch_size = 16  # Reduce batch size to fit in GPU memory

print("Starting model training")
try:
    history = model.fit(
        [x_train[0], x_train[1], x_train[2]], 
        {'final_output_age': y_train_age, 'final_output_gender': y_train_gender, 'final_output_race': y_train_race_onehot},
        validation_data=(
            [x_val[0], x_val[1], x_val[2]], 
            {'final_output_age': y_val_age, 'final_output_gender': y_val_gender, 'final_output_race': y_val_race_onehot}
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, lr_scheduler, checkpoint]
    )
except Exception as e:
    print(f"An error occurred during training: {e}")
    print("Exiting.")
    exit(1)

print("Model training completed successfully")

# Save the final model
model.save(os.path.join(checkpoint_dir, 'age_gender_race_model_final.keras'))

print("\nFinal Training and Validation Metrics:")
print(f"Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Training Age MAE: {history.history['final_output_age_mae'][-1]:.4f}")
print(f"Validation Age MAE: {history.history['val_final_output_age_mae'][-1]:.4f}")
print(f"Training Gender Accuracy: {history.history['final_output_gender_accuracy'][-1]:.4f}")
print(f"Validation Gender Accuracy: {history.history['val_final_output_gender_accuracy'][-1]:.4f}")
print(f"Training Race Accuracy: {history.history['final_output_race_accuracy'][-1]:.4f}")
print(f"Validation Race Accuracy: {history.history['val_final_output_race_accuracy'][-1]:.4f}")

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(history.history['final_output_age_mae'], label='Training Age MAE')
plt.plot(history.history['val_final_output_age_mae'], label='Validation Age MAE')
plt.title('Age Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history.history['final_output_gender_accuracy'], label='Training Gender Accuracy')
plt.plot(history.history['val_final_output_gender_accuracy'], label='Validation Gender Accuracy')
plt.title('Gender Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(history.history['final_output_race_accuracy'], label='Training Race Accuracy')
plt.plot(history.history['val_final_output_race_accuracy'], label='Validation Race Accuracy')
plt.title('Race Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics_updated.png')
print("Training metrics plot saved as 'training_metrics_updated.png'")

plt.show(block=False)
plt.pause(120)  # Display the plot for 2 minutes
plt.close()

print("Training metrics plot closed after 2 minutes.")
