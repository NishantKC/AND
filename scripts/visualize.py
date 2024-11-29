import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocess import load_training_data

model = load_model("models/unet_model.h5", compile=False)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# Load model and data
test_images, test_masks = load_training_data("dataset/train/images/", "dataset/train/masks/")

# Make predictions
predictions = model.predict(test_images)

# Visualize a sample image, mask, and prediction
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(test_images[0].squeeze(), cmap="gray")
plt.subplot(1, 3, 2)
plt.title("Ground Truth")
plt.imshow(test_masks[0].squeeze(), cmap="gray")
plt.subplot(1, 3, 3)
plt.title("Prediction")
plt.imshow(predictions[0].squeeze(), cmap="gray")
plt.show()