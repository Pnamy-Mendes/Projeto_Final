import os
import numpy as np
import cv2
import yaml
import dlib
from skimage.color import rgb2lab

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_tensorflow_gpu():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)

    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")

def detect_landmarks(image, predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) > 0:
        shape = predictor(gray, rects[0])
        landmarks = np.zeros((68, 2), dtype=int)
        for i in range(68):
            landmarks[i] = (shape.part(i).x, shape.part(i).y)
        return landmarks
    return np.zeros((68, 2), dtype=int)

def extract_hair_color(image, landmarks):
    if landmarks is None or len(landmarks) == 0:
        return np.full(3, -1)
    hair_region = image[:landmarks[0][1], :]
    if hair_region.size == 0:
        return np.full(3, -1)
    hair_region = cv2.cvtColor(hair_region, cv2.COLOR_BGR2RGB)
    hair_region = rgb2lab(hair_region)
    mean_color = hair_region.mean(axis=(0, 1))
    return mean_color

def extract_face_structure(landmarks):
    if landmarks is None or len(landmarks) == 0:
        return np.full(5, -1)
    jawline = landmarks[0:17]
    left_eyebrow = landmarks[17:22]
    right_eyebrow = landmarks[22:27]
    nose_bridge = landmarks[27:31]
    nose_tip = landmarks[31:36]
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    mouth_outer = landmarks[48:60]
    mouth_inner = landmarks[60:68]
    face_height = np.linalg.norm(nose_bridge[0] - jawline[8])
    face_width = np.linalg.norm(jawline[0] - jawline[-1])
    eye_distance = np.linalg.norm(left_eye[0] - right_eye[3])
    mouth_width = np.linalg.norm(mouth_outer[0] - mouth_outer[6])
    nose_width = np.linalg.norm(nose_tip[0] - nose_tip[4])
    eye_to_nose_ratio = eye_distance / nose_width if nose_width != 0 else -1
    return np.array([face_height, face_width, eye_distance, mouth_width, eye_to_nose_ratio])

def extract_additional_features(image, landmarks):
    features = []
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    features.append(laplacian_var)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    mask = cv2.inRange(image, lower_black, upper_black)
    facial_hair_density = np.sum(mask) / (image.shape[0] * image.shape[1])
    features.append(facial_hair_density)
    eyebrow_widths = []
    eyebrow_heights = []
    for eyebrow in [landmarks[17:22], landmarks[22:27]]:
        x_coords = eyebrow[:, 0]
        y_coords = eyebrow[:, 1]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        eyebrow_widths.append(width)
        eyebrow_heights.append(height)
    features.extend(eyebrow_widths)
    features.extend(eyebrow_heights)
    target_length = 10
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    elif len(features) > target_length:
        features = features[:target_length]
    return np.array(features)

def extract_symmetry(landmarks):
    if landmarks is None or len(landmarks) == 0:
        return -1
    left_side = landmarks[:34]
    right_side = landmarks[34:]
    symmetry_score = np.mean(np.abs(left_side - right_side))
    return symmetry_score

def extract_skin_texture(image, landmarks):
    if landmarks is None or len(landmarks) == 0:
        return -1
    skin_region = image[landmarks[0:27, 1].min():landmarks[8, 1].max(), landmarks[0:17, 0].min():landmarks[16, 0].max()]
    if skin_region.size == 0:
        return -1
    gray_skin = cv2.cvtColor(skin_region, cv2.COLOR_BGR2GRAY)
    texture_score = cv2.Laplacian(gray_skin, cv2.CV_64F).var()
    return texture_score

def extract_features(image_path, config):
    try:
        predictor_path = config['datasets']['predictor_path']
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        landmarks = detect_landmarks(image, predictor_path)
        hair_color = extract_hair_color(image, landmarks)
        face_structure = extract_face_structure(landmarks)
        additional_features = extract_additional_features(image, landmarks)
        symmetry = extract_symmetry(landmarks)
        skin_texture = extract_skin_texture(image, landmarks)
        features = np.concatenate([hair_color, face_structure, additional_features, [symmetry, skin_texture]], axis=0)
        
        # Ensure features array has a consistent length
        target_length = 14  # Adjusted target length
        if len(features) < target_length:
            features = np.pad(features, (0, target_length - len(features)), 'constant', constant_values=-1)
        elif len(features) > target_length:
            features = features[:target_length]

        return image, landmarks, features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros((256, 256, 3), dtype=np.float32), np.zeros((68, 2), dtype=np.float32), np.full(14, -1)

def parse_label_from_filename(filename):
    try:
        age, gender, race, _ = filename.split('_')
        return int(age), int(gender), int(race)
    except Exception as e:
        print(f"Error parsing label from filename {filename}: {e}")
        return -1, -1, -1

def generate_cache(data_dir, config, cache_path, max_images=None):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.jpg')]
    if max_images:
        image_paths = image_paths[:max_images]

    images = []
    features = []
    ages = []
    genders = []
    races = []

    cached_image_paths = []
    cached_data = []

    print(f"Starting cache generation for {len(image_paths)} images")

    for idx, image_path in enumerate(image_paths):
        try:
            print(f"Processing image {idx + 1}/{len(image_paths)}: {image_path}")
            image, _, combined_features = extract_features(image_path, config)
            age, gender, race = parse_label_from_filename(os.path.basename(image_path))

            if len(combined_features) == 14:  # Ensure correct feature length
                images.append(image)
                features.append(combined_features)
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

                if idx > 0 and idx % 100 == 0:
                    np.savez_compressed(cache_path, image_paths=np.array(cached_image_paths), data=np.array(cached_data, dtype=object))
                    print(f"Cache updated at {cache_path}")
                    print(f"Processed {idx + 1}/{len(image_paths)} images")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    np.savez_compressed(cache_path, image_paths=np.array(cached_image_paths), data=np.array(cached_data, dtype=object))
    print(f"Final cache saved at {cache_path}")
    print(f"Total images processed: {len(image_paths)}")



def main():
    # Load configuration
    config = load_config('config.yaml')

    # Define paths
    data_dir = config['datasets']['age_gender_race_data']
    cache_path = os.path.join('cache', 'UTK_age_gender_race', 'split_cache', 'data_cache.npz')

    # Generate cache
    generate_cache(data_dir, config, cache_path, max_images=5000)

if __name__ == "__main__":
    main()
