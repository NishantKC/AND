import os
import cv2
import numpy as np

def load_training_data(image_dir, mask_dir, target_size=(256, 256)):
    images = []
    masks = []

    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(image_files, mask_files):
        # Load and check the image
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load image {img_path}. Skipping.")
            continue

        # Resize and normalize the image
        img_resized = cv2.resize(img, target_size) / 255.0
        images.append(img_resized)

        # Load and check the mask
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not load mask {mask_path}. Skipping.")
            continue

        # Resize and normalize the mask
        mask_resized = cv2.resize(mask, target_size) / 255.0
        masks.append(mask_resized)

    # Convert lists to NumPy arrays
    images = np.expand_dims(np.array(images), axis=-1)
    masks = np.expand_dims(np.array(masks), axis=-1)

    return images, masks