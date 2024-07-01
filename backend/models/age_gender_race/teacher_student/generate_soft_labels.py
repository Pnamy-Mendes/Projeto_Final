# backend/models/age_gender_race/teacher_student/generate_soft_labels.py
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import yaml
from tqdm import tqdm

# Adjust the backend directory import
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(backend_dir)

def load_config(config_file):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return None

def setup_tensorflow_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)

def data_loader(data_dir, validation_split, input_shape, config, max_images=None, use_cache_only=True):
    try:
        cache_dir = os.path.join(backend_dir, 'cache', 'UTK_age_gender_race')
        cache_path = os.path.join(cache_dir, 'data_cache.npz')
        
        if use_cache_only and os.path.exists(cache_path):
            print("Loading data from cache...")
            data = np.load(cache_path, allow_pickle=True)
            x_train, y_train, x_val, y_val = data['x_train'], data['y_train'], data['x_val'], data['y_val']
            return (x_train, y_train), (x_val, y_val)
        
        # If not using cache, you should add the code to load and preprocess the data here
        # For now, we'll raise an exception to indicate that only cache loading is supported in this script
        raise NotImplementedError("Data loading from original dataset is not implemented in this script.")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None

def main():
    # Load configuration
    config = load_config(os.path.join(backend_dir, 'config.yaml'))
    if config is None:
        print("Failed to load configuration. Exiting.")
        return

    setup_tensorflow_gpu()

    # Load data
    data_dir = config['datasets']['age_gender_race_data']
    validation_split = config['validation_split']
    input_shape = config['model_params']['input_shape']
    max_images = 1000
    use_cache_only = True  # Load data from cache

    train_data, _ = data_loader(data_dir, validation_split, input_shape, config, max_images=max_images, use_cache_only=use_cache_only)
    if train_data is None:
        print("Failed to load training data. Exiting.")
        return
    
    x_train, y_train = train_data

    model_path = os.path.join(backend_dir, 'models', 'age_gender_race', 'models', 'teacher_model_best.keras')
    if not os.path.exists(model_path):
        print(f"Teacher model not found at {model_path}. Exiting.")
        return

    teacher_model = load_model(model_path)
    print("Teacher model loaded successfully.")

    # Generate soft labels using the teacher model
    soft_labels = teacher_model.predict([x_train[0], x_train[1], x_train[2]], batch_size=32)
    print("Soft labels generated successfully.")

    # Save the soft labels to disk
    cache_dir = os.path.join(backend_dir, 'cache', 'UTK_age_gender_race')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    soft_labels_path = os.path.join(cache_dir, 'soft_labels.npz')
    np.savez(soft_labels_path, soft_labels=soft_labels)
    print(f"Soft labels saved to {soft_labels_path}.")

if __name__ == "__main__":
    main()
