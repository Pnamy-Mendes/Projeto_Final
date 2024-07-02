import os
import numpy as np

# Function to split cache into multiple smaller caches
def split_cache(cache_path, output_dir, max_images_per_cache=1000):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found at {cache_path}")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load existing cache
    cached_data = np.load(cache_path, allow_pickle=True)
    cached_image_paths = cached_data['image_paths']
    cached_images = cached_data['images']
    cached_landmarks = cached_data['landmarks']
    cached_features = cached_data['features']
    cached_ages = cached_data['ages']
    cached_genders = cached_data['genders']
    cached_races = cached_data['races']

    # Determine the number of new cache files needed
    num_cache_files = len(cached_image_paths) // max_images_per_cache
    if len(cached_image_paths) % max_images_per_cache != 0:
        num_cache_files += 1

    # Split and save the cache files
    for i in range(num_cache_files):
        start_idx = i * max_images_per_cache
        end_idx = min((i + 1) * max_images_per_cache, len(cached_image_paths))

        split_image_paths = cached_image_paths[start_idx:end_idx]
        split_images = cached_images[start_idx:end_idx]
        split_landmarks = cached_landmarks[start_idx:end_idx]
        split_features = cached_features[start_idx:end_idx]
        split_ages = cached_ages[start_idx:end_idx]
        split_genders = cached_genders[start_idx:end_idx]
        split_races = cached_races[start_idx:end_idx]

        split_cache_path = os.path.join(output_dir, f"data_cache_{i + 1}.npz")
        np.savez_compressed(split_cache_path, image_paths=split_image_paths, images=split_images,
                            landmarks=split_landmarks, features=split_features,
                            ages=split_ages, genders=split_genders, races=split_races)
        print(f"Saved split cache to {split_cache_path}")

    print("Cache splitting completed successfully.")

# Example usage
cache_path = 'cache/UTK_age_gender_race/split_cache/data_cache_1.npz'  # Path to the existing cache file
output_dir = 'cache/UTK_age_gender_race/split_cache'    # Directory to save the split cache files
split_cache(cache_path, output_dir)
