import os
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define paths
teacher_model_path = 'models/mood/teacher_model.keras'
train_data_dir = config['datasets']['fer_train']
val_data_dir = config['datasets']['fer_validation']
features_train_path = 'models/mood/features_train.npy'
features_val_path = 'models/mood/features_val.npy'
labels_train_path = 'models/mood/labels_train.npy'
labels_val_path = 'models/mood/labels_val.npy'

# Load the teacher model
teacher_model = load_model(teacher_model_path)

# Extract the layer output up to the penultimate layer
feature_extractor = Model(inputs=teacher_model.input, outputs=teacher_model.layers[-2].output)

# Data augmentation
datagen = ImageDataGenerator(rescale=1./255)

# Image data generators
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

validation_generator = datagen.flow_from_directory(
    val_data_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

# Extract features and labels
train_features = feature_extractor.predict(train_generator)
val_features = feature_extractor.predict(validation_generator)

train_labels = train_generator.classes
val_labels = validation_generator.classes

# Save features and labels
np.save(features_train_path, train_features)
np.save(features_val_path, val_features)
np.save(labels_train_path, train_labels)
np.save(labels_val_path, val_labels)

print(f'Train features shape: {train_features.shape}')
print(f'Validation features shape: {val_features.shape}')
print(f'Train labels shape: {train_labels.shape}')
print(f'Validation labels shape: {val_labels.shape}')
