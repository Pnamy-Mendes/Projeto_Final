import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import yaml

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define paths
data_dir = config['datasets']['fer_train']
val_data_dir = config['datasets']['fer_validation']
cache_dir = 'models/mood'
final_model_save_path = os.path.join(cache_dir, 'final_trained_model_with_features.keras')

# Function to save features and labels to cache
def save_cache(train_features, val_features, train_labels, val_labels):
    np.save(os.path.join(cache_dir, 'features_train.npy'), train_features)
    np.save(os.path.join(cache_dir, 'features_val.npy'), val_features)
    np.save(os.path.join(cache_dir, 'labels_train.npy'), train_labels)
    np.save(os.path.join(cache_dir, 'labels_val.npy'), val_labels)

# Try to load the cache
try:
    train_features = np.load(os.path.join(cache_dir, 'features_train.npy'))
    val_features = np.load(os.path.join(cache_dir, 'features_val.npy'))
    train_labels = np.load(os.path.join(cache_dir, 'labels_train.npy'))
    val_labels = np.load(os.path.join(cache_dir, 'labels_val.npy'))
    print("Cache loaded successfully.")
except FileNotFoundError:
    print("Cache not found. Extracting features from images.")
    # Define image data generators
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(48, 48),
        batch_size=64,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(48, 48),
        batch_size=64,
        class_mode='categorical',
        subset='validation'
    )

    # Load the VGG16 model pre-trained on ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(7, activation='softmax')(x)

    # Create the full model
    teacher_model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the teacher model
    teacher_model.fit(train_generator, epochs=100, validation_data=validation_generator)

    # Save the teacher model
    teacher_model.save(os.path.join(cache_dir, 'teacher_model.keras'))

    # Extract the layer output up to the penultimate layer
    feature_extractor = Model(inputs=teacher_model.input, outputs=teacher_model.layers[-2].output)

    # Extract features and labels
    train_features = feature_extractor.predict(train_generator)
    val_features = feature_extractor.predict(validation_generator)
    train_labels = train_generator.classes
    val_labels = validation_generator.classes

    # Save features and labels to cache
    save_cache(train_features, val_features, train_labels, val_labels)
    print("Features and labels extracted and saved to cache.")

# Convert labels to one-hot encoding
num_classes = config['model_params']['n_classes']
train_labels = to_categorical(train_labels, num_classes=num_classes)
val_labels = to_categorical(val_labels, num_classes=num_classes)

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
student_model.compile(optimizer='adam',
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

# Save the trained model architecture as an image
from tensorflow.keras.utils import plot_model

plot_model(student_model, to_file='student_model.png', show_shapes=True, show_layer_names=True)
print("Model architecture saved as 'student_model.png'.")
