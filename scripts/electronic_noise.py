# scripts/electronic_noise.py

import os
import numpy as np
from PIL import Image

def add_electronic_noise(image_array, noise_level=0.05):
    """
    Add electronic noise (Gaussian additive noise) to an image array.
    """
    noise = np.random.normal(0, noise_level, image_array.shape)
    noisy_image = image_array + noise
    return np.clip(noisy_image, 0, 1)

def process_electronic_noise(input_dir, output_dir, noise_level=0.05):
    """
    Apply electronic noise to all images in the input directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image = Image.open(os.path.join(input_dir, filename)).convert("L")
            image_array = np.array(image, dtype=np.float32) / 255.0
            noisy_image_array = add_electronic_noise(image_array, noise_level=noise_level)
            noisy_image = (noisy_image_array * 255).astype(np.uint8)
            noisy_image_pil = Image.fromarray(noisy_image)
            noisy_image_pil.save(os.path.join(output_dir, filename.replace(".", "_electronic.")))

if __name__ == "__main__":
    process_electronic_noise("../input_images", "../output_images/electronic_noise", noise_level=0.05)