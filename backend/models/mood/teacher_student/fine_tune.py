import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import yaml

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define paths
train_data_dir = config['datasets']['fer_train']
val_data_dir = config['datasets']['fer_validation']
model_save_path = './models/mood/final_trained_model_with_features.keras'
initial_model_path = './models/mood/final_trained_model_initial.keras'

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Image data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical'
)

validation_generator = train_datagen.flow_from_directory(
    val_data_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical'
)

# Load the initially trained model
original_model = load_model(initial_model_path)

# Ensure all layers are trainable
for layer in original_model.layers:
    layer.trainable = True

# Define a function to generate additional features
def generate_dummy_features(batch_size, feature_shape):
    while True:
        dummy_features = np.random.rand(batch_size, *feature_shape).astype(np.float32)
        yield dummy_features

# Create a new generator that provides both image data and dummy feature data
def combined_generator(image_generator, feature_generator):
    while True:
        image_batch, labels = next(image_generator)
        feature_batch = next(feature_generator)
        # Ensure feature_batch has the same batch size as image_batch
        if len(feature_batch) != len(image_batch):
            feature_batch = np.tile(feature_batch, (len(image_batch) // len(feature_batch) + 1, 1))[:len(image_batch)]
        yield (image_batch, feature_batch), labels

# Create the feature generators
input_shape_features = tuple(config['model_params']['input_shape_features'])
train_feature_generator = generate_dummy_features(64, input_shape_features)
validation_feature_generator = generate_dummy_features(64, input_shape_features)

# Combine the generators
train_combined_generator = combined_generator(train_generator, train_feature_generator)
validation_combined_generator = combined_generator(validation_generator, validation_feature_generator)

# Create TensorFlow dataset from combined generator
def generator_to_tf_dataset(generator, batch_size, img_shape, feat_shape, n_classes):
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (
                tf.TensorSpec(shape=(batch_size, *img_shape), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, *feat_shape), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(batch_size, n_classes), dtype=tf.float32)
        )
    )
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Define the dataset shapes and number of classes
input_shape_image = config['model_params']['input_shape_image']
n_classes = config['model_params']['n_classes']

train_dataset = generator_to_tf_dataset(lambda: train_combined_generator, 64, (48, 48, 3), input_shape_features, n_classes)
validation_dataset = generator_to_tf_dataset(lambda: validation_combined_generator, 64, (48, 48, 3), input_shape_features, n_classes)

# Define callbacks
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Fine-tune the model
original_model.fit(
    train_dataset,
    epochs=100,
    steps_per_epoch=train_generator.samples // 64,
    validation_data=validation_dataset,
    validation_steps=validation_generator.samples // 64,
    callbacks=[checkpoint, early_stopping]
)

print("Fine-tuning completed.")
