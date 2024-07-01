import os
import numpy as np
import cv2
import logging

def load_data(data_dir):
    images = []
    ages = []
    genders = []
    races = []
    landmarks = []  # To store landmarks

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def print_memory_usage():
        import psutil
        memory_info = psutil.virtual_memory()
        total_memory = memory_info.total / (1024 ** 3)
        available_memory = memory_info.available / (1024 ** 3)
        used_memory = total_memory - available_memory
        memory_usage = used_memory / total_memory * 100
        print(f"Total memory: {total_memory:.2f} GB")
        print(f"Available memory: {available_memory:.2f} GB")
        print(f"Used memory: {used_memory:.2f} GB")
        print(f"Memory usage: {memory_usage:.1f}%")

    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            try:
                # Parse the filename to extract labels
                parts = filename.split('_')
                age = int(parts[0])
                gender = int(parts[1])
                if len(parts) > 2 and parts[2].isdigit():
                    race = int(parts[2])
                else:
                    logging.error(f"Filename {filename} does not contain a valid race part")
                    continue
                file_path = os.path.join(data_dir, filename)

                # Load the image
                image = cv2.imread(file_path)
                image = cv2.resize(image, (256, 256))
                images.append(image)
                ages.append(age)
                genders.append(gender)
                races.append(race)

                # For simplicity, generating random landmarks here
                # Replace this part with actual landmark extraction logic
                landmark = np.random.rand(68, 2) * 256  # Fake landmarks
                landmarks.append(landmark)

                if len(images) % 1000 == 0:
                    print_memory_usage()
                    logging.debug(f"Processed {len(images)} images")

            except Exception as e:
                logging.error(f"Error loading image {filename}: {e}")

    return np.array(images), np.array(ages), np.array(genders), np.array(races), np.array(landmarks)
