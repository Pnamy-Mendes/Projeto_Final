import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import yaml

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define paths
features_train_path = 'models/mood/features_train.npy'
features_val_path = 'models/mood/features_val.npy'
labels_train_path = 'models/mood/labels_train.npy'
labels_val_path = 'models/mood/labels_val.npy'
final_model_save_path = 'models/mood/final_trained_model_with_features.keras'

# Load features and labels
train_features = np.load(features_train_path)
val_features = np.load(features_val_path)
train_labels = np.load(labels_train_path)
val_labels = np.load(labels_val_path)

# Convert labels to one-hot encoding
num_classes = config['model_params']['n_classes']
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=num_classes)

# Create the student model
input_shape_features = train_features.shape[1:]

input_features = Input(shape=input_shape_features)
x = Flatten()(input_features)
x = Dense(512)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)
x = Dense(256)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

student_model = Model(inputs=input_features, outputs=x)

# Compile the model
student_model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint(final_model_save_path, monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Train the model
student_model.fit(
    train_features, train_labels,
    epochs=100,
    batch_size=64,
    validation_data=(val_features, val_labels),
    callbacks=[checkpoint, early_stopping]
)

print("Training completed.")
