import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Concatenate, Dense, Flatten, Input, Dropout, BatchNormalization, 
                                     GlobalAveragePooling2D, Conv2D, MaxPooling2D, LeakyReLU)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.mixed_precision import global_policy, set_global_policy
import yaml
import keras_tuner as kt
from sklearn.model_selection import train_test_split
import cv2
import visualkeras
from tensorflow.keras.utils import plot_model 
import dlib
from skimage.color import rgb2lab
import time
import logging

# Load and set up configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

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
        
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")

class PrintAgePredictions(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict([self.validation_data[0], self.validation_data[1], self.validation_data[2]])
        age_predictions = predictions[0]
        print(f"Age predictions at epoch {epoch + 1}: {age_predictions}")


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

def detect_beard_mustache(image, landmarks):
    if landmarks is None or len(landmarks) == 0:
        return -1, -1

    beard_area = image[landmarks[48][1]:landmarks[8][1], landmarks[48][0]:landmarks[54][0]]
    mustache_area = image[landmarks[33][1]:landmarks[51][1], landmarks[31][0]:landmarks[35][0]]

    def has_facial_hair(area):
        if area.size == 0:
            return -1
        gray_area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        _, binary_area = cv2.threshold(gray_area, 60, 255, cv2.THRESH_BINARY_INV)
        return np.mean(binary_area)

    beard_presence = has_facial_hair(beard_area)
    mustache_presence = has_facial_hair(mustache_area)

    return beard_presence, mustache_presence

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

    # Wrinkle detection using Laplacian variance
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    features.append(laplacian_var)

    # Crow's feet detection around eyes
    left_eye_region = gray_image[landmarks[36:42, 1].min():landmarks[36:42, 1].max(), landmarks[36:42, 0].min():landmarks[36:42, 0].max()]
    right_eye_region = gray_image[landmarks[42:48, 1].min():landmarks[42:48, 1].max(), landmarks[42:48, 0].min():landmarks[42:48, 0].max()]
    crows_feet_left = cv2.Laplacian(left_eye_region, cv2.CV_64F).var() if left_eye_region.size > 0 else 0
    crows_feet_right = cv2.Laplacian(right_eye_region, cv2.CV_64F).var() if right_eye_region.size > 0 else 0
    features.append(crows_feet_left)
    features.append(crows_feet_right)

    # Smile lines detection around mouth
    mouth_region = gray_image[landmarks[48:60, 1].min():landmarks[48:60, 1].max(), landmarks[48:60, 0].min():landmarks[48:60, 0].max()]
    smile_lines = cv2.Laplacian(mouth_region, cv2.CV_64F).var() if mouth_region.size > 0 else 0
    features.append(smile_lines)

    # Eye Aspect Ratio (EAR)
    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    ear_left = eye_aspect_ratio(left_eye)
    ear_right = eye_aspect_ratio(right_eye)
    features.append(ear_left)
    features.append(ear_right)

    # Face proportions
    face_height = np.linalg.norm(landmarks[27] - landmarks[8])
    face_width = np.linalg.norm(landmarks[0] - landmarks[16])
    features.append(face_height)
    features.append(face_width)

    # Ensure all features have consistent length
    target_length = 14
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


def extract_features(image_path, config, target='age'):
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

        beard_presence, mustache_presence = detect_beard_mustache(image, landmarks)
        facial_hair_features = np.array([beard_presence, mustache_presence])

        if target == 'age':
            features = np.concatenate([hair_color, face_structure, additional_features, [symmetry, skin_texture], facial_hair_features], axis=0)
        elif target == 'gender':
            features = np.concatenate([face_structure, additional_features, facial_hair_features], axis=0)
        else:
            raise ValueError(f"Unknown target: {target}")

        # Ensure features have consistent length of 14
        target_length = 14
        if len(features) < target_length:
            features = np.pad(features, (0, target_length - len(features)), 'constant', constant_values=-1)
        elif len(features) > target_length:
            features = features[:target_length]

        # Check for NaN or infinite values and replace them with -1
        features = np.nan_to_num(features, nan=-1.0, posinf=-1.0, neginf=-1.0)
        
        return image, landmarks, features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros((256, 256, 3), dtype=np.float32), np.zeros((68, 2), dtype=np.float32), np.full(14, -1)


def parse_label_from_filename(filename):
    try:
        parts = filename.split('_')
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
        return age, gender, race
    except Exception as e:
        print(f"Error parsing label from filename {filename}: {e}")
        return -1, -1, -1


def data_loader(data_dir, validation_split, input_shape, config, batch_size=4, max_images=None, use_cache_only=False, target='age'):
    cache_dir = os.path.join('cache', 'UTK_age_gender_race', 'split_cache')
    os.makedirs(cache_dir, exist_ok=True)

    images = []
    features = []
    ages = []
    genders = []
    races = []

    cache_path = os.path.join(cache_dir, f'data_cache_{target}.npz')
    cached_data = []

    logging.debug(f"Data directory: {data_dir}")
    logging.debug(f"Predictor path: {config['datasets']['predictor_path']}")
    logging.debug(f"Max images: {max_images}")

    if os.path.exists(cache_path) and use_cache_only:
        print(f"Loading data from {cache_path}...")
        cached = np.load(cache_path, allow_pickle=True)
        cached_image_paths = cached['image_paths'].tolist()
        cached_data = cached['data'].tolist()

        for data in cached_data:
            images.append(data['image'])
            features.append(np.nan_to_num(data['features'], nan=-1.0, posinf=-1.0, neginf=-1.0))
            ages.append(data['age'])
            genders.append(data['gender'])
            races.append(data['race'])

        if max_images:
            images = images[:max_images]
            features = features[:max_images]
            ages = ages[:max_images]
            genders = genders[:max_images]
            races = races[:max_images]

    else:
        image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.jpg')]
        if max_images:
            image_paths = image_paths[:max_images]

        print(f"Starting to load {len(image_paths)} images from dataset...")

        predictor_path = config['datasets']['predictor_path']
        start_time = time.time()

        for idx, image_path in enumerate(image_paths):
            try:
                print(f"Starting feature extraction for image {idx + 1}/{len(image_paths)}")
                image, _, combined_features = extract_features(image_path, config, target)
                print(f"Finished feature extraction for image {idx + 1}/{len(image_paths)}")

                age, gender, race = parse_label_from_filename(os.path.basename(image_path))

                if len(combined_features) == 14:  # Ensure correct feature length
                    # Ensure the image is resized to the expected input shape
                    resized_image = cv2.resize(image, (input_shape[0], input_shape[1]))
                    images.append(resized_image)
                    features.append(combined_features)
                    ages.append(age)
                    genders.append(gender)
                    races.append(race)

                    cached_data.append({
                        'image': resized_image,
                        'features': combined_features,
                        'age': age,
                        'gender': gender,
                        'race': race
                    })

                    if idx > 0 and idx % 250 == 0:
                        np.savez_compressed(cache_path, image_paths=np.array(image_paths), data=np.array(cached_data, dtype=object))
                        print(f"Processed {idx} images; Cache saved at {cache_path}")

                    # Progress and ETA
                    if idx > 0 and idx % 50 == 0:
                        elapsed_time = time.time() - start_time
                        avg_time_per_image = elapsed_time / idx
                        eta = avg_time_per_image * (len(image_paths) - idx)
                        print(f"{idx}/{len(image_paths)} images processed, {elapsed_time:.2f} seconds elapsed, ETA: {eta:.2f} seconds")

            except Exception as e:
                print(f"Error extracting features for {image_path}: {e}")

        np.savez_compressed(cache_path, image_paths=np.array(image_paths), data=np.array(cached_data, dtype=object))
        print(f"Final cache saved at {cache_path}")

    if len(images) == 0 or len(features) == 0:
        print(f"Error: No images or features loaded. Images: {len(images)}, Features: {len(features)}")
        return None, None

    # Ensure all features have consistent shape
    max_feature_length = max(len(f) for f in features)
    features = np.array([np.pad(f, (0, max_feature_length - len(f)), 'constant', constant_values=-1) for f in features], dtype=np.float32)

    # Ensure all images are of the same shape
    print("Checking image shapes before conversion to NumPy array...")
    for idx, img in enumerate(images):
        print(f"Image {idx} shape: {img.shape}")

    # Convert images to NumPy array with consistent shape
    images = np.array(images, dtype=np.float32)

    ages = np.array(ages, dtype=np.int32)
    genders = np.array(genders, dtype=np.int32)
    races = np.array(races, dtype=np.int32)

    images = images / 255.0

    random_state = 42  # Define random state for reproducibility

    try:
        x_train_img, x_val_img, y_train_ages, y_val_ages = train_test_split(images, ages, test_size=validation_split, random_state=random_state)
        x_train_features, x_val_features = train_test_split(features, test_size=validation_split, random_state=random_state)
        y_train_genders, y_val_genders = train_test_split(genders, test_size=validation_split, random_state=random_state)
        y_train_races, y_val_races = train_test_split(races, test_size=validation_split, random_state=random_state)
    except ValueError as e:
        print(f"Error during train_test_split: {e}")
        raise

    # One-hot encode the race labels for the model training process
    y_train_race_onehot = tf.keras.utils.to_categorical(y_train_races, num_classes=5)
    y_val_race_onehot = tf.keras.utils.to_categorical(y_val_races, num_classes=5)

    print(f"Training images shape: {x_train_img.shape}")
    print(f"Validation images shape: {x_val_img.shape}")
    print(f"Training features shape: {x_train_features.shape}")
    print(f"Validation features shape: {x_val_features.shape}")
    print(f"Training ages shape: {y_train_ages.shape}")
    print(f"Validation ages shape: {y_val_ages.shape}")
    print(f"Training genders shape: {y_train_genders.shape}")
    print(f"Validation genders shape: {y_val_genders.shape}")
    print(f"Training races shape: {y_train_races.shape}")
    print(f"Validation races shape: {y_val_races.shape}")

    train_data = (x_train_img, x_train_features, y_train_ages, y_train_genders, y_train_races, y_train_race_onehot)
    val_data = (x_val_img, x_val_features, y_val_ages, y_val_genders, y_val_races, y_val_race_onehot)

    return train_data, val_data

def hyperparameter_tuning(x_train_img, x_train_features, y_train_age, y_train_gender, y_train_race, x_val_img, x_val_features, y_val_age, y_val_gender, y_val_race):
    try:
        # Hyperparameter Tuning for Age Output
        print("Starting hyperparameter tuning for age output")
        stop_early_age = EarlyStopping(monitor='val_mae', patience=2, mode='min')
        tuner_age = kt.Hyperband(
            lambda hp: build_model(hp, 'age'),
            objective='val_mae',
            max_epochs=5,
            factor=2,
            directory='hyperband',
            project_name='age_tuning',
            executions_per_trial=1
        )
        # Reduce batch size to fit GPU memory
        batch_size = 4  # Adjust as needed to fit memory constraints
        tuner_age.search([x_train_img, np.zeros((len(x_train_img), 68, 2)), x_train_features], y_train_age, epochs=5, validation_data=([x_val_img, np.zeros((len(x_val_img), 68, 2)), x_val_features], y_val_age), batch_size=batch_size, callbacks=[stop_early_age])
        best_hps_age = tuner_age.get_best_hyperparameters(num_trials=1)[0]
        print(f"Hyperparameter tuning for age completed.")

        # Hyperparameter Tuning for Gender Output
        print("Starting hyperparameter tuning for gender output")
        stop_early_gender = EarlyStopping(monitor='val_accuracy', patience=2, mode='max')
        tuner_gender = kt.Hyperband(
            lambda hp: build_model(hp, 'gender'),
            objective='val_accuracy',
            max_epochs=5,
            factor=2,
            directory='hyperband',
            project_name='gender_tuning',
            executions_per_trial=1
        )
        # Reduce batch size to fit GPU memory
        tuner_gender.search([x_train_img, np.zeros((len(x_train_img), 68, 2)), x_train_features], y_train_gender, epochs=5, validation_data=([x_val_img, np.zeros((len(x_val_img), 68, 2)), x_val_features], y_val_gender), batch_size=batch_size, callbacks=[stop_early_gender])
        best_hps_gender = tuner_gender.get_best_hyperparameters(num_trials=1)[0]
        print(f"Hyperparameter tuning for gender completed.")

        # Hyperparameter Tuning for Race Output
        print("Starting hyperparameter tuning for race output")
        stop_early_race = EarlyStopping(monitor='val_accuracy', patience=2, mode='max')
        tuner_race = kt.Hyperband(
            lambda hp: build_model(hp, 'race'),
            objective='val_accuracy',
            max_epochs=5,
            factor=2,
            directory='hyperband',
            project_name='race_tuning',
            executions_per_trial=1
        )

        # One-hot encode the race labels for the hyperparameter tuning process
        y_train_race_onehot = tf.keras.utils.to_categorical(y_train_race, num_classes=5)
        y_val_race_onehot = tf.keras.utils.to_categorical(y_val_race, num_classes=5)

        # Reduce batch size to fit GPU memory
        tuner_race.search([x_train_img, np.zeros((len(x_train_img), 68, 2)), x_train_features], y_train_race_onehot, epochs=5, validation_data=([x_val_img, np.zeros((len(x_val_img), 68, 2)), x_val_features], y_val_race_onehot), batch_size=batch_size, callbacks=[stop_early_race])
        best_hps_race = tuner_race.get_best_hyperparameters(num_trials=1)[0]
        print(f"Hyperparameter tuning for race completed.")

        return best_hps_age, best_hps_gender, best_hps_race
    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")
        raise

def build_model(hp, output_type=None):
    input_shape_images = (256, 256, 3)
    input_shape_landmarks = (68, 2)
    input_shape_features = (14,)
    num_races = 5

    image_input = Input(shape=input_shape_images, name='image_input')
    landmark_input = Input(shape=input_shape_landmarks, name='landmark_input')
    features_input = Input(shape=input_shape_features, name='features_input')

    x = Conv2D(hp.Int('conv_1_filter', min_value=32, max_value=256, step=32), (3, 3), padding='same')(image_input)
    x = LeakyReLU(negative_slope=0.1)(x)  # Updated here
    x = BatchNormalization()(x)
    x = Conv2D(hp.Int('conv_2_filter', min_value=32, max_value=256, step=32), (3, 3), padding='same')(x)
    x = LeakyReLU(negative_slope=0.1)(x)  # Updated here
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(hp.Int('conv_3_filter', min_value=32, max_value=256, step=32), (3, 3), padding='same')(x)
    x = LeakyReLU(negative_slope=0.1)(x)  # Updated here
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = GlobalAveragePooling2D()(x)

    flat_landmarks = Flatten()(landmark_input)
    concatenated_features = Concatenate(name='concatenate_features')([x, flat_landmarks, features_input])

    dense_concat = Dense(hp.Int('dense_units', min_value=128, max_value=256, step=32), activation='relu', name='dense_concat')(concatenated_features)
    dropout = Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1), name='dropout_concat')(dense_concat)
    batch_norm = BatchNormalization(name='batch_norm_concat')(dropout)

    if output_type == 'age':
        final_output = Dense(1, name='final_output_age', dtype=tf.float32)(batch_norm)
        loss = 'mean_squared_error'
        metrics = ['mae']
    elif output_type == 'gender':
        final_output = Dense(1, activation='sigmoid', name='final_output_gender', dtype=tf.float32)(batch_norm)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    elif output_type == 'race':
        final_output = Dense(num_races, activation='softmax', name='final_output_race', dtype=tf.float32)(batch_norm)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:
        raise ValueError("output_type must be 'age', 'gender', or 'race'")

    model = Model(inputs=[image_input, landmark_input, features_input], outputs=final_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss=loss,
        metrics=metrics
    )
    return model


# Prepare final training data
train_data, val_data = data_loader(
    config['datasets']['age_gender_race_data'], 
    config['validation_split'], 
    config['model_params']['input_shape'], 
    config, 
    batch_size=4, 
    max_images=128, 
    use_cache_only=False
)
(x_train_img, x_train_features, y_train_age, y_train_gender, y_train_race, y_train_race_onehot) = train_data
(x_val_img, x_val_features, y_val_age, y_val_gender, y_val_race, y_val_race_onehot) = val_data

# Perform hyperparameter tuning
best_hps_age, best_hps_gender, best_hps_race = hyperparameter_tuning(x_train_img, x_train_features, y_train_age, y_train_gender, y_train_race, x_val_img, x_val_features, y_val_age, y_val_gender, y_val_race)



# Combine the best hyperparameters
# Define the custom metric function
def clip_age(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 0, 116)
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# Build the model function
def build_age_gender_race_model(input_shape_images, input_shape_landmarks, input_shape_features, num_races):
    try:
        image_input = Input(shape=input_shape_images, name='image_input')
        landmark_input = Input(shape=input_shape_landmarks, name='landmark_input')
        features_input = Input(shape=input_shape_features, name='features_input')

        x = Conv2D(best_hps_age.get('conv_1_filter'), (3, 3), padding='same')(image_input)
        x = LeakyReLU(negative_slope=0.1)(x)  # Updated here
        x = BatchNormalization()(x)
        x = Conv2D(best_hps_age.get('conv_2_filter'), (3, 3), padding='same')(x)
        x = LeakyReLU(negative_slope=0.1)(x)  # Updated here
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(best_hps_age.get('conv_3_filter'), (3, 3), padding='same')(x)
        x = LeakyReLU(negative_slope=0.1)(x)  # Updated here
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = GlobalAveragePooling2D()(x)

        flat_landmarks = Flatten()(landmark_input)
        concatenated_features = Concatenate(name='concatenate_features')([x, flat_landmarks, features_input])

        dense_concat = Dense(best_hps_age.get('dense_units'), activation='relu', name='dense_concat')(concatenated_features)
        dropout = Dropout(best_hps_age.get('dropout'), name='dropout_concat')(dense_concat)
        batch_norm = BatchNormalization(name='batch_norm_concat')(dropout)

        final_output_age = Dense(1, name='final_output_age', dtype=tf.float32)(batch_norm)
        final_output_gender = Dense(1, activation='sigmoid', name='final_output_gender', dtype=tf.float32)(batch_norm)
        final_output_race = Dense(num_races, activation='softmax', name='final_output_race', dtype=tf.float32)(batch_norm)

        model = Model(inputs=[image_input, landmark_input, features_input], outputs=[final_output_age, final_output_gender, final_output_race])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps_age.get('learning_rate')),
            loss={
                'final_output_age': 'mean_squared_error',
                'final_output_gender': 'binary_crossentropy',
                'final_output_race': 'categorical_crossentropy'
            },
            metrics={
                'final_output_age': 'mae',
                'final_output_gender': 'accuracy',
                'final_output_race': 'accuracy'
            }
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps_age.get('learning_rate')),
            loss={
                'final_output_age': 'mean_squared_error',
                'final_output_gender': 'binary_crossentropy',
                'final_output_race': 'categorical_crossentropy'
            },
            metrics={
                'final_output_age': clip_age,
                'final_output_gender': 'accuracy',
                'final_output_race': 'accuracy'
            }
        )

        return model
    except Exception as e:
        print(f"Error building the model: {e}")
        raise


# Build and train the final model
model = build_age_gender_race_model(
    input_shape_images=(256, 256, 3),
    input_shape_landmarks=(68, 2),
    input_shape_features=(14,),
    num_races=5
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
lr_scheduler = LearningRateScheduler(lambda epoch: 1e-4 * 0.95 ** epoch)
checkpoint_dir = os.path.join('models', 'age_gender_race', 'models')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, 'age_gender_race_model_best.keras'), monitor='val_loss', save_best_only=True)

epochs = 100
batch_size = 4  # Reduce batch size to fit in GPU memory

print_age_predictions = PrintAgePredictions()

print("Starting model training")
history = model.fit(
    [x_train_img, np.zeros((len(x_train_img), 68, 2)), x_train_features], 
    {'final_output_age': y_train_age, 'final_output_gender': y_train_gender, 'final_output_race': y_train_race_onehot},
    validation_data=(
        [x_val_img, np.zeros((len(x_val_img), 68, 2)), x_val_features], 
        {'final_output_age': y_val_age, 'final_output_gender': y_val_gender, 'final_output_race': y_val_race_onehot}
    ),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping, reduce_lr, lr_scheduler, checkpoint, print_age_predictions]
)

print("Model training completed successfully")

# Save the final model
model.save(os.path.join(checkpoint_dir, 'age_gender_race_model_final.keras'))

print("\nFinal Training and Validation Metrics:")
print(f"Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Training Age MAE: {history.history['final_output_age_mae'][-1]:.4f}")
print(f"Validation Age MAE: {history.history['val_final_output_age_mae'][-1]:.4f}")
print(f"Training Gender Accuracy: {history.history['final_output_gender_accuracy'][-1]:.4f}")
print(f"Validation Gender Accuracy: {history.history['val_final_output_gender_accuracy'][-1]:.4f}")
print(f"Training Race Accuracy: {history.history['final_output_race_accuracy'][-1]:.4f}")
print(f"Validation Race Accuracy: {history.history['val_final_output_race_accuracy'][-1]:.4f}")

# Plot training metrics
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(history.history['final_output_age_mae'], label='Training Age MAE')
plt.plot(history.history['val_final_output_age_mae'], label='Validation Age MAE')
plt.title('Age Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history.history['final_output_gender_accuracy'], label='Training Gender Accuracy')
plt.plot(history.history['val_final_output_gender_accuracy'], label='Validation Gender Accuracy')
plt.title('Gender Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(history.history['final_output_race_accuracy'], label='Training Race Accuracy')
plt.plot(history.history['val_final_output_race_accuracy'], label='Validation Race Accuracy')
plt.title('Race Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics_updated.png')
print("Training metrics plot saved as 'training_metrics_updated.png'")

plt.show(block=False)
plt.pause(120)  # Display the plot for 2 minutes
plt.close()

print("Training metrics plot closed after 2 minutes.")

# Visualize the model architecture
plot_model(model, to_file='agr_model_architecture.png', show_shapes=True, show_layer_names=True)
print("Model architecture plot saved as 'model_architecture.png'")

# Visualize using visualkeras
visualkeras.layered_view(model, to_file='agr_model_visualization.png').show()
print("Model visualization saved as 'model_visualization.png'")


# Evaluate the model
print("Evaluating the model...")
evaluation_results = model.evaluate(
    [x_val_img, np.zeros((len(x_val_img), 68, 2)), x_val_features], 
    {'final_output_age': y_val_age, 'final_output_gender': y_val_gender, 'final_output_race': tf.keras.utils.to_categorical(y_val_race, num_classes=5)}
)
print("Evaluation results:", evaluation_results)

# Example predictions
print("Making example predictions...")
example_predictions = model.predict([x_val_img[:10], np.zeros((10, 68, 2)), x_val_features[:10]])
print("Example predictions:", example_predictions)