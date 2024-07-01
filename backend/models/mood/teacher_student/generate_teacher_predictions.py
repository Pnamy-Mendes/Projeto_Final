import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load configuration
import yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define paths
data_dir = config['datasets']['fer_train']

# Define image data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=False  # Important to maintain the order for feature extraction
)

# Load the trained teacher model
teacher_model_path = './models/mood/teacher_model.keras'
teacher_model = load_model(teacher_model_path)

# Generate predictions
predictions = teacher_model.predict(train_generator, verbose=1)

# Save the predictions and corresponding file paths
np.save('./teacher_student/teacher_predictions.npy', predictions)
np.save('./teacher_student/filenames.npy', train_generator.filenames)
