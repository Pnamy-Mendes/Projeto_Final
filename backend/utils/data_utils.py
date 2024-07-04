import numpy as np
import os
import logging
from utils.feature_extraction_mood import extract_features

def save_cache(train_features, val_features, train_labels, val_labels, cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    np.save(os.path.join(cache_dir, 'features_train.npy'), train_features)
    np.save(os.path.join(cache_dir, 'features_val.npy'), val_features)
    np.save(os.path.join(cache_dir, 'labels_train.npy'), train_labels)
    np.save(os.path.join(cache_dir, 'labels_val.npy'), val_labels)

def extract_label_from_path(image_path):
    label = os.path.basename(os.path.dirname(image_path))
    label_map = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
    return label_map.get(label, -1)

def load_fer_data(data_dir, predictor_path, cache_dir, split='train', cache_only=False):
    features_list = []
    labels_list = []

    for idx, (root, _, files) in enumerate(os.walk(data_dir)):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(root, file)
                try:
                    logging.info(f"Processing image: {image_path}")
                    image, landmarks, features = extract_features(image_path, predictor_path)
                    label = extract_label_from_path(image_path)
                    if label == -1:
                        logging.error(f"Label not found for {image_path}")
                        continue
                    if len(features) == 0:
                        logging.error(f"No features extracted for {image_path}")
                        continue
                    features_list.append(features)
                    labels_list.append(label)
                    logging.debug(f"Extracted features for {image_path}: {features}")
                except Exception as e:
                    logging.error(f"Error processing {image_path}: {e}")

        if idx > 0 and idx % 100 == 0 and not cache_only:
            logging.info(f"Processed {idx} files, saving intermediate cache.")
            features_array = np.array(features_list)
            labels_array = np.array(labels_list)
            np.save(os.path.join(cache_dir, f'features_{split}.npy'), features_array)
            np.save(os.path.join(cache_dir, f'labels_{split}.npy'), labels_array)

    if not features_list:
        raise ValueError("No features extracted. Check your dataset and extraction function.")
        
    max_length = max(len(f) for f in features_list)
    features_list = [np.pad(f, (0, max_length - len(f)), 'constant', constant_values=0) if len(f) < max_length else f for f in features_list]

    return np.array(features_list), np.array(labels_list)
