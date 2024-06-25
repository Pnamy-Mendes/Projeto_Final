import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Dropout, BatchNormalization, LeakyReLU, Attention, Reshape, Add
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import yaml

# Load configuration
with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define paths
train_data_dir = config['datasets']['fer_train']
val_data_dir = config['datasets']['fer_validation']
teacher_model_path = '../models/mood/teacher_model.keras'
initial_model_save_path = '../models/mood/final_trained_model_initial.keras'
final_model_save_path = '../models/mood/final_trained_model_with_features.keras'

# Load the teacher model
teacher_model = load_model(teacher_model_path)

# Extract the last layer's output
teacher_output = teacher_model.layers[-2].output  # Assuming the last layer is the Softmax layer

# Create a new model combining the teacher model's features and additional layers
input_shape_image = (48, 48, 3)
input_shape_features = tuple(config['model_params']['input_shape_features'])
n_classes = config['model_params']['n_classes']

input_image = Input(shape=input_shape_image)
input_features = Input(shape=input_shape_features)

x1 = teacher_model(input_image)
x1 = Flatten()(x1)

x2 = Flatten()(input_features)

combined = Concatenate()([x1, x2])
combined = Dense(512)(combined)
combined = BatchNormalization()(combined)
combined = LeakyReLU()(combined)
combined = Dropout(0.5)(combined)
combined = Dense(256)(combined)
combined = BatchNormalization()(combined)
combined = LeakyReLU()(combined)
combined = Dropout(0.5)(combined)
combined = Dense(n_classes, activation='softmax')(combined)

original_model = Model(inputs=[input_image, input_features], outputs=combined)

# Compile the model
original_model.compile(optimizer=Adam(learning_rate=0.001),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

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

val_datagen = ImageDataGenerator(rescale=1./255)

# Image data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical'
)

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
        yield ([image_batch, feature_batch], labels)

# Create the feature generators
train_feature_generator = generate_dummy_features(64, input_shape_features)
validation_feature_generator = generate_dummy_features(64, input_shape_features)

# Combine the generators
train_combined_generator = combined_generator(train_generator, train_feature_generator)
validation_combined_generator = combined_generator(validation_generator, validation_feature_generator)

# Define callbacks
checkpoint = ModelCheckpoint(final_model_save_path, monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Train the model
original_model.fit(
    train_combined_generator,
    epochs=100,
    steps_per_epoch=train_generator.samples // 64,
    validation_data=validation_combined_generator,
    validation_steps=validation_generator.samples // 64,
    callbacks=[checkpoint, early_stopping]
)

# Save the initial model before fine-tuning
original_model.save(initial_model_save_path)

print("Training completed.")
