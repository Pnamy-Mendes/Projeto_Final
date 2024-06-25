import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score

# Load configuration
import yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define paths
data_dir = config['datasets']['fer_validation']
model_path = './models/mood/teacher_model.keras'
#model_path = './models/mood/final_trained_model_with_features.keras'

# Load the teacher model
teacher_model = load_model(model_path)

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
print("Classification Report:\n")
print(classification_report(labels, predicted_labels, target_names=list(mood_label_map.values())))

# Calculate and print accuracy per mood
accuracy_per_mood = {}
for mood, idx in mood_label_map.items():
    mask = labels == idx
    accuracy = accuracy_score(labels[mask], predicted_labels[mask])
    accuracy_per_mood[mood] = accuracy

print("\nAccuracy per Mood:")
for mood, accuracy in accuracy_per_mood.items():
    print(f"{mood}: {accuracy:.2f}")

# Print each prediction and the correct mood
print("\nPredictions:")
for i in range(len(predicted_labels)):
    true_mood = mood_label_map[labels[i]]
    predicted_mood = mood_label_map[predicted_labels[i]]
    print(f"Image {i + 1}: Predicted Label: {predicted_mood}, True Label: {true_mood}")
