import os
import cv2

def validate_images(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Invalid or unreadable image: {filepath}")

# Validate both images and masks
validate_images("dataset/train/images/")
validate_images("dataset/train/masks/")