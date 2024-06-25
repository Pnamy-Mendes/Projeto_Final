import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from teacher_student.data_utils import load_fer_data
from models.mood.mood_model import get_advanced_model, add_attention_layer

# Load configuration
import yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define paths
data_dir = config['datasets']['fer_train']
predictor_path = config['datasets']['predictor_path']
max_images_per_class = config['datasets'].get('max_images_per_class', None)

# Load FER dataset and extracted features
features, _ = load_fer_data(data_dir, predictor_path, max_images_per_class)

# Load teacher predictions
teacher_predictions = np.load('./teacher_student/teacher_predictions.npy')
filenames = np.load('./teacher_student/filenames.npy')

# Ensure the order of features matches the order of filenames
sorted_indices = np.argsort(filenames)
features = features[sorted_indices]
teacher_predictions = teacher_predictions[sorted_indices]

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(features, teacher_predictions, test_size=0.2, random_state=42)

# Define and train your original model using teacher predictions as labels
input_shape_features = X_train.shape[1:]
original_model = get_advanced_model(
    input_shape_features=input_shape_features,
    leaky_relu_slope=0.1,
    dropout_rate=0.5,
    regularization_rate=0.01,
    n_classes=7,
    dense_units_1=512,
    dense_units_2=256,
    dense_units_3=128,
    dense_units_4=64,
    dense_units_5=32
)

original_model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val))

# Save the updated original model
original_model.save('./models/mood/student_model.keras')
