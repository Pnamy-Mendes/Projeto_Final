import numpy as np
import cv2
import dlib
from skimage.color import rgb2lab
import logging

logging.basicConfig(level=logging.DEBUG)

def detect_landmarks(image, predictor_path):
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            landmarks = np.zeros((68, 2), dtype="int")
            for i in range(0, 68):
                landmarks[i] = (shape.part(i).x, shape.part(i).y)
            return landmarks
    except Exception as e:
        logging.error(f"Error detecting landmarks: {e}")
        return np.zeros((68, 2), dtype="int")  # Return empty landmarks on error
    return np.zeros((68, 2), dtype="int")  # Return empty landmarks if no face is detected

def extract_hair_color(image, landmarks):
    if landmarks is None or len(landmarks) == 0:
        return np.full(3, -1)  # Default to special value if landmarks are not detected
    try:
        hair_region = image[landmarks[17:27, 1].min():landmarks[0:17, 1].max(), landmarks[17:27, 0].min():landmarks[0:17, 0].max()]
        if hair_region.size == 0:
            return np.full(3, -1)  # Default to special value if hair region is empty
        hair_region = cv2.cvtColor(hair_region, cv2.COLOR_BGR2RGB)
        hair_region = rgb2lab(hair_region)
        mean_color = hair_region.mean(axis=(0, 1))
        return mean_color  # L, a, b values
    except Exception as e:
        logging.error(f"Error extracting hair color: {e}")
        return np.full(3, -1)

def extract_face_structure(landmarks):
    if landmarks is None or len(landmarks) == 0:
        return np.full(5, -1)  # Default to special value if landmarks are not detected
    try:
        jawline = landmarks[0:17]
        left_eyebrow = landmarks[17:22]
        right_eyebrow = landmarks[22:27]
        nose_bridge = landmarks[27:31]
        nose_tip = landmarks[31:36]
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        mouth_outer = landmarks[48:60]
        mouth_inner = landmarks[60:68]

        # Compute distances and ratios as features
        face_height = np.linalg.norm(nose_bridge[0] - jawline[8])
        face_width = np.linalg.norm(jawline[0] - jawline[-1])
        eye_distance = np.linalg.norm(left_eye[0] - right_eye[3])
        mouth_width = np.linalg.norm(mouth_outer[0] - mouth_outer[6])
        nose_width = np.linalg.norm(nose_tip[0] - nose_tip[4])
        eye_to_nose_ratio = eye_distance / nose_width if nose_width != 0 else -1  # Avoid division by zero
        
        return np.array([face_height, face_width, eye_distance, mouth_width, eye_to_nose_ratio])
    except Exception as e:
        logging.error(f"Error extracting face structure: {e}")
        return np.full(5, -1)

def extract_symmetry(landmarks):
    if landmarks is None or len(landmarks) == 0:
        return -1  # Default to special value if landmarks are not detected
    try:
        left_side = landmarks[:34]
        right_side = landmarks[34:]
        symmetry_score = np.mean(np.abs(left_side - right_side))
        return symmetry_score
    except Exception as e:
        logging.error(f"Error extracting symmetry: {e}")
        return -1

def extract_skin_texture(image, landmarks):
    if landmarks is None or len(landmarks) == 0:
        return -1  # Default to special value if landmarks are not detected
    try:
        skin_region = image[landmarks[0:27, 1].min():landmarks[8, 1].max(), landmarks[0:17, 0].min():landmarks[16, 0].max()]
        if skin_region.size == 0:
            return -1  # Default to special value if skin region is empty
        gray_skin = cv2.cvtColor(skin_region, cv2.COLOR_BGR2GRAY)
        texture_score = cv2.Laplacian(gray_skin, cv2.CV_64F).var()
        return texture_score
    except Exception as e:
        logging.error(f"Error extracting skin texture: {e}")
        return -1

def extract_additional_features(image, landmarks):
    features = []

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

    # Ensure the length of features is consistent
    return np.array(features)

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
        
        logging.debug(f"Extracted features for {image_path}: {features}")
        return image, landmarks, features
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return np.zeros((256, 256, 3), dtype=np.float32), np.zeros((68, 2), dtype=np.float32), np.full(13, -1)
