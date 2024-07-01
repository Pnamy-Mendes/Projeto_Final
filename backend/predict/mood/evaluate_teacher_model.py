import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score
import yaml
import logging

# Set up logging
logging.basicConfig(filename='project.log', level=logging.DEBUG)

# Load configuration
try:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    logging.debug(f'Configuration loaded: {config}')
except FileNotFoundError as e:
    logging.error(f'FileNotFoundError: {e}')
    raise

# Define paths
data_dir = config['datasets']['fer_validation']
model_path = './models/mood/models/teacher_model.keras'

# Check if the model file exists
if not os.path.exists(model_path):
    logging.error(f'FileNotFoundError: Model file not found at {model_path}')
    raise FileNotFoundError(f'Model file not found at {model_path}')

# Load the teacher model
try:
    teacher_model = load_model(model_path)
    logging.info(f'Model loaded successfully from {model_path}')
except Exception as e:
    logging.error(f'Error loading model: {e}')
    raise

# Print model summary
teacher_model.summary(print_fn=logging.info)
print("Model Summary:")
teacher_model.summary()

# Print model layers
print("\nModel Layers:")
for i, layer in enumerate(teacher_model.layers):
    output_shape = layer.output_shape if hasattr(layer, 'output_shape') else 'No output shape'
    print(f"Layer {i}: {layer.name}, Output Shape: {output_shape}, Parameters: {layer.count_params()}")
    logging.info(f"Layer {i}: {layer.name}, Output Shape: {output_shape}, Parameters: {layer.count_params()}")

# Define image data generator for normalization
datagen = ImageDataGenerator(rescale=1./255)

# Load validation data
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)

# Get the ground truth labels
labels = validation_generator.classes
class_indices = validation_generator.class_indices
class_indices = {v: k for k, v in class_indices.items()}

# Make predictions
predictions = teacher_model.predict(validation_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Define mood label map
mood_label_map = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}

# Print classification report
classification_report_str = classification_report(labels, predicted_labels, target_names=list(mood_label_map.values()))
print("Classification Report:\n")
print(classification_report_str)
logging.info(f"Classification Report:\n{classification_report_str}")

# Calculate and print accuracy per mood
accuracy_per_mood = {}
for mood, idx in mood_label_map.items():
    mask = labels == idx
    accuracy = accuracy_score(labels[mask], predicted_labels[mask])
    accuracy_per_mood[mood] = accuracy

print("\nAccuracy per Mood:")
for mood, accuracy in accuracy_per_mood.items():
    print(f"{mood}: {accuracy:.2f}")
    logging.info(f"{mood}: {accuracy:.2f}")

# Print each prediction and the correct mood
print("\nPredictions:")
for i in range(len(predicted_labels)):
    true_mood = mood_label_map[labels[i]]
    predicted_mood = mood_label_map[predicted_labels[i]]
    print(f"Image {i + 1}: Predicted Label: {predicted_mood}, True Label: {true_mood}")
    logging.info(f"Image {i + 1}: Predicted Label: {predicted_mood}, True Label: {true_mood}")
