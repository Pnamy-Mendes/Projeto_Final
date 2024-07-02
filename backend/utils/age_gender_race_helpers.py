# age_gender_race_helpers.py
import yaml
import tensorflow as tf
import numpy as np
import cv2
import os
import psutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.feature_extraction import detect_landmarks, extract_hair_color, extract_face_structure, extract_additional_features

def load_config(config_file):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        raise

def setup_tensorflow_gpu():
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            except RuntimeError as e:
                print(e)
    except Exception as e:
        print(f"Error setting up TensorFlow GPU: {e}")

def load_image(image_path, target_size):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        image = cv2.resize(image, target_size)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        raise

def parse_label_from_filename(filename):
    try:
        age, gender, race, _ = filename.split('_')
        return int(age), int(gender), int(race)
    except Exception as e:
        print(f"Error parsing label from filename {filename}: {e}")
        return -1, -1, -1

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: RSS={memory_info.rss / 1024 ** 2:.2f} MB, VMS={memory_info.vms / 1024 ** 2:.2f} MB")

def load_split_cache(cache_dir, max_files=None):
    try:
        cache_files = sorted([f for f in os.listdir(cache_dir) if f.startswith('data_cache_') and f.endswith('.npz')])

        if max_files:
            cache_files = cache_files[:max_files]

        all_cached_image_paths = []
        all_cached_images = []
        all_cached_landmarks = []
        all_cached_features = []
        all_cached_ages = []
        all_cached_genders = []
        all_cached_races = []

        for cache_file in cache_files:
            cache_path = os.path.join(cache_dir, cache_file)
            print(f"Loading data from {cache_path}...")
            log_memory_usage()
            cached_data = np.load(cache_path, allow_pickle=True)
            all_cached_image_paths.extend(cached_data['image_paths'].tolist())
            all_cached_images.extend(cached_data['images'].tolist())
            if 'landmarks' in cached_data:
                all_cached_landmarks.extend(cached_data['landmarks'].tolist())
            if 'features' in cached_data:
                all_cached_features.extend(cached_data['features'].tolist())
            if 'ages' in cached_data:
                all_cached_ages.extend(cached_data['ages'].tolist())
            if 'genders' in cached_data:
                all_cached_genders.extend(cached_data['genders'].tolist())
            if 'races' in cached_data:
                all_cached_races.extend(cached_data['races'].tolist())
            del cached_data  # Clear memory

        return all_cached_image_paths, all_cached_images, all_cached_landmarks, all_cached_features, all_cached_ages, all_cached_genders, all_cached_races
    except Exception as e:
        print(f"Error loading split cache from {cache_dir}: {e}")
        raise

def save_cache(cache_dir, split_idx, cached_image_paths, cached_images, cached_landmarks, cached_features, cached_ages, cached_genders, cached_races):
    split_cache_path = os.path.join(cache_dir, f'data_cache_{split_idx}.npz')
    np.savez_compressed(split_cache_path, image_paths=np.array(cached_image_paths), images=np.array(cached_images, dtype=np.float32),
                        landmarks=np.array(cached_landmarks, dtype=np.float32), features=np.array(cached_features, dtype=np.float32),
                        ages=np.array(cached_ages, dtype=np.int32), genders=np.array(cached_genders, dtype=np.int32), races=np.array(cached_races, dtype=np.int32))
    print(f"Cache saved at {split_cache_path}")

def data_loader(data_dir, validation_split, input_shape, config, batch_size, max_images=None, use_cache_only=False):
    try:
        cache_dir = os.path.join('cache', 'UTK_age_gender_race', 'split_cache')
        os.makedirs(cache_dir, exist_ok=True)

        if use_cache_only:
            cached_image_paths, cached_images, cached_landmarks, cached_features, cached_ages, cached_genders, cached_races = load_split_cache(cache_dir, max_files=2)
        else:
            cached_image_paths = []
            cached_images = []
            cached_landmarks = []
            cached_features = []
            cached_ages = []
            cached_genders = []
            cached_races = []

        images = []
        landmarks = []
        features = []
        ages = []
        genders = []
        races = []

        image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.jpg')]
        if max_images:
            image_paths = image_paths[:max_images]

        predictor_path = config['datasets']['predictor_path']
        split_idx = len(cached_image_paths) // 1000

        for idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc='Loading images and extracting features'):
            try:
                if image_path in cached_image_paths:
                    print(f"Loading {image_path} from cache.")
                    index = cached_image_paths.index(image_path)
                    images.append(cached_images[index])
                    landmarks.append(cached_landmarks[index])
                    features.append(cached_features[index])
                    ages.append(cached_ages[index])
                    genders.append(cached_genders[index])
                    races.append(cached_races[index])
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
                    landmarks.append(landmark)
                    features.append(combined_features)
                    age, gender, race = parse_label_from_filename(os.path.basename(image_path))
                    ages.append(age)
                    genders.append(gender)
                    races.append(race)

                    cached_image_paths.append(image_path)
                    cached_images.append(image)
                    cached_landmarks.append(landmark)
                    cached_features.append(combined_features)
                    cached_ages.append(age)
                    cached_genders.append(gender)
                    cached_races.append(race)

                    if len(cached_image_paths) % 250 == 0:
                        save_cache(cache_dir, split_idx, cached_image_paths, cached_images, cached_landmarks, cached_features, cached_ages, cached_genders, cached_races)
                        if len(cached_image_paths) >= 1000:
                            split_idx += 1
                            cached_image_paths = []
                            cached_images = []
                            cached_landmarks = []
                            cached_features = []
                            cached_ages = []
                            cached_genders = []
                            cached_races = []

            except Exception as e:
                print(f"Error extracting features for {image_path}: {e}")

        # Ensure consistent lengths by filtering out any mismatched entries
        min_length = min(len(images), len(landmarks), len(features), len(ages), len(genders), len(races))
        images = images[:min_length]
        landmarks = landmarks[:min_length]
        features = features[:min_length]
        ages = ages[:min_length]
        genders = genders[:min_length]
        races = races[:min_length]

        images = np.array(images, dtype=np.float32)  # Ensure images are float32
        landmarks = np.array(landmarks, dtype=np.float32)  # Use dtype=float32 for homogeneous arrays
        features = np.array(features, dtype=np.float32)  # Ensure features are float32
        ages = np.array(ages, dtype=np.int32)  # Ensure labels are int32
        genders = np.array(genders, dtype=np.int32)
        races = np.array(races, dtype=np.int32)

        images = images / 255.0  # Normalize images

        x_train_img, x_val_img, x_train_landmarks, x_val_landmarks, x_train_features, x_val_features, y_train_age, y_val_age, y_train_gender, y_val_gender, y_train_race, y_val_race = train_test_split(
            images, landmarks, features, ages, genders, races, test_size=validation_split, random_state=42
        )

        x_train = [np.array(x_train_img), np.array(x_train_landmarks, dtype=np.float32), np.array(x_train_features)]
        y_train = [np.array(y_train_age), np.array(y_train_gender), np.array(y_train_race)]
        x_val = [np.array(x_val_img), np.array(x_val_landmarks, dtype=np.float32), np.array(x_val_features)]
        y_val = [np.array(y_val_age), np.array(y_val_gender), np.array(y_val_race)]

        # Save updated cache
        if cached_image_paths:
            save_cache(cache_dir, split_idx, cached_image_paths, cached_images, cached_landmarks, cached_features, cached_ages, cached_genders, cached_races)

        train_gen = data_generator(x_train, y_train, batch_size)
        val_gen = data_generator(x_val, y_val, batch_size)
        steps_per_epoch_train = len(x_train[0]) // batch_size
        steps_per_epoch_val = len(x_val[0]) // batch_size

        return train_gen, val_gen, steps_per_epoch_train, steps_per_epoch_val
    except Exception as e:
        print(f"Error in data_loader: {e}")
        raise

def data_generator(x, y, batch_size):
    try:
        num_samples = len(x[0])
        while True:
            for offset in range(0, num_samples, batch_size):
                x_batch = [np.array(feat[offset:offset + batch_size]) for feat in x]
                y_batch = [np.array(label[offset:offset + batch_size]) for label in y]
                yield x_batch, y_batch
    except Exception as e:
        print(f"Error in data_generator: {e}")
        raise

def extract_additional_features(image, landmarks):
    features = []

    try:
        # Ensure image depth is supported
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Wrinkles Detection using Laplacian Variance
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        features.append(laplacian_var)

        # Facial Hair Detection using color and edge detection
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])
        mask = cv2.inRange(image, lower_black, upper_black)
        facial_hair_density = np.sum(mask) / (image.shape[0] * image.shape[1])
        features.append(facial_hair_density)

        # Eyebrow characteristics (width and height)
        eyebrow_widths = []
        eyebrow_heights = []

        for eyebrow in [landmarks[17:22], landmarks[22:27]]:  # Left and right eyebrow landmarks
            x_coords = eyebrow[:, 0]
            y_coords = eyebrow[:, 1]
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            eyebrow_widths.append(width)
            eyebrow_heights.append(height)

        features.extend(eyebrow_widths)
        features.extend(eyebrow_heights)
    except cv2.error as e:
        print(f"OpenCV error in extract_additional_features: {e}")
        features = np.array([np.nan] * 13)  # Return NaN for each feature if error occurs

    return np.array(features)

def detect_landmarks(image, predictor_path):
    try:
        import dlib
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        if len(rects) == 0:
            return None

        landmarks = []
        for rect in rects:
            shape = predictor(gray, rect)
            for i in range(0, 68):
                landmarks.append((shape.part(i).x, shape.part(i).y))

        return np.array(landmarks)
    except Exception as e:
        print(f"Error detecting landmarks: {e}")
        return None

def extract_hair_color(image, landmarks):
    try:
        if landmarks.shape != (68, 2):
            return np.array([-1, -1, -1])
        hair_region = landmarks[0:17]  # Assuming the hair region is above the forehead landmarks
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hair_region, 1)
        hair_pixels = cv2.bitwise_and(image, image, mask=mask)
        hair_color = cv2.mean(hair_pixels, mask=mask)[:3]  # Average color in BGR
        return np.array(hair_color)
    except Exception as e:
        print(f"Error extracting hair color: {e}")
        return np.array([-1, -1, -1])

def extract_face_structure(landmarks):
    try:
        if landmarks.shape != (68, 2):
            return np.array([-1, -1, -1, -1])
        face_length = np.linalg.norm(landmarks[8] - landmarks[19])
        face_width = np.linalg.norm(landmarks[0] - landmarks[16])
        jaw_width = np.linalg.norm(landmarks[4] - landmarks[12])
        face_aspect_ratio = face_length / face_width

        return np.array([face_length, face_width, jaw_width, face_aspect_ratio])
    except Exception as e:
        print(f"Error extracting face structure: {e}")
        return np.array([-1, -1, -1, -1])
