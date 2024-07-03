import yaml
import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.feature_extraction import detect_landmarks, extract_hair_color, extract_face_structure, extract_additional_features

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

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

def load_image(image_path, target_size):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    image = cv2.resize(image, target_size)
    return image

def parse_label_from_filename(filename):
    try:
        age, gender, race, _ = filename.split('_')
        return int(age), int(gender), int(race)
    except Exception as e:
        print(f"Error parsing label from filename {filename}: {e}")
        return -1, -1, -1

def load_split_cache(cache_dir):
    cached_image_paths = []
    cached_data = []

    cache_files = sorted([f for f in os.listdir(cache_dir) if f.startswith('data_cache_') and f.endswith('.npz')])

    for cache_file in cache_files:
        cache_path = os.path.join(cache_dir, cache_file)
        print(f"Loading data from {cache_path}...")
        cached = np.load(cache_path, allow_pickle=True)
        cached_image_paths.extend(cached['image_paths'].tolist())
        cached_data.extend(cached['data'].tolist())

    return cached_image_paths, cached_data

def data_loader(data_dir, validation_split, input_shape, config, batch_size, max_images=None, use_cache_only=False):
    cache_dir = os.path.join('cache', 'UTK_age_gender_race', 'split_cache')
    os.makedirs(cache_dir, exist_ok=True)

    if use_cache_only:
        cached_image_paths, cached_data = load_split_cache(cache_dir)
    else:
        cached_image_paths = []
        cached_data = []

    images = []
    features = []
    ages = []
    genders = []
    races = []

    image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.jpg')]
    if max_images:
        image_paths = image_paths[:max_images]

    predictor_path = config['datasets']['predictor_path']

    for idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc='Loading images and extracting features'):
        try:
            if image_path in cached_image_paths:
                print(f"Loading {image_path} from cache.")
                index = cached_image_paths.index(image_path)
                data = cached_data[index]
                images.append(data['image'])
                features.append(data['features'])
                ages.append(data['age'])
                genders.append(data['gender'])
                races.append(data['race'])
            else:
                image = load_image(image_path, input_shape[:2])
                landmark = detect_landmarks(image, predictor_path)
                
                if landmark is None or len(landmark) == 0:
                    print(f"No landmarks detected for image {image_path}. Skipping...")
                    continue
                
                hair_color = extract_hair_color(image, landmark)
                face_structure = extract_face_structure(landmark)
                additional_features = extract_additional_features(image, landmark)
                combined_features = np.concatenate([hair_color, face_structure, additional_features], axis=0)
                print(f"Extracted features for {image_path}: {combined_features}")

                images.append(image)
                features.append(combined_features)
                age, gender, race = parse_label_from_filename(os.path.basename(image_path))
                ages.append(age)
                genders.append(gender)
                races.append(race)

                cached_image_paths.append(image_path)
                cached_data.append({
                    'image': image,
                    'features': combined_features,
                    'age': age,
                    'gender': gender,
                    'race': race
                })

                if idx > 0 and idx % 250 == 0:
                    split_idx = len(cached_image_paths) // 1000
                    split_cache_path = os.path.join(cache_dir, f'data_cache_{split_idx}.npz')
                    np.savez_compressed(split_cache_path, image_paths=np.array(cached_image_paths), data=np.array(cached_data, dtype=object))
                    print(f"Cache saved at {split_cache_path}")

                    # Clear lists to save memory
                    cached_image_paths = cached_image_paths[-1000:]
                    cached_data = cached_data[-1000:]

        except Exception as e:
            print(f"Error extracting features for {image_path}: {e}")

    # Ensure consistent lengths by filtering out any mismatched entries
    min_length = min(len(images), len(features), len(ages), len(genders), len(races))
    images = images[:min_length]
    features = features[:min_length]
    ages = ages[:min_length]
    genders = genders[:min_length]
    races = races[:min_length]

    images = np.array(images, dtype=np.float32)  # Ensure images are float32
    features = np.array(features, dtype=np.float32)  # Ensure features are float32
    ages = np.array(ages, dtype=np.int32)  # Ensure labels are int32
    genders = np.array(genders, dtype=np.int32)
    races = np.array(races, dtype=np.int32)

    images = images / 255.0  # Normalize images

    # Ensure combined features have consistent shapes
    combined_features = []
    for img, feat in zip(images, features):
        combined = np.concatenate([img.flatten(), feat])
        combined_features.append(combined)

    combined_features = np.array(combined_features, dtype=np.float32)

    combined_outputs = np.column_stack((ages, genders, races))

    print("Shapes of datasets before splitting:")
    print(f"Combined Inputs: {combined_features.shape}")
    print(f"Combined Outputs: {combined_outputs.shape}")

    x_train, x_val, y_train, y_val = train_test_split(
        combined_features, combined_outputs, test_size=validation_split, random_state=42
    )

    print("Shapes of datasets after splitting:")
    print(f"x_train: {x_train.shape}")
    print(f"x_val: {x_val.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_val: {y_val.shape}")

    # Save updated cache
    if len(cached_image_paths) > 0:
        split_idx = len(cached_image_paths) // 1000
        split_cache_path = os.path.join(cache_dir, f'data_cache_{split_idx}.npz')
        np.savez_compressed(split_cache_path, image_paths=np.array(cached_image_paths), data=np.array(cached_data, dtype=object))
        print(f"Final cache saved at {split_cache_path}")

    return x_train, x_val, y_train, y_val
