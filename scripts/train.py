from tensorflow.keras.optimizers import Adam
from preprocess import load_training_data
from model import unet_model

# Load training data
train_images, train_masks = load_training_data(
    "dataset/train/images",
    "dataset/train/masks"
)

# Define and compile the model
model = unet_model(input_shape=(256, 256, 1))
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(train_images, train_masks, epochs=20, batch_size=16)

# Save the trained model
model.save("models/unet_model.h5")