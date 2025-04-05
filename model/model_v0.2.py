import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ----------------------------------------
# 1. Load images and assign labels
# ----------------------------------------
data_dir = "merge"  # 모든 이미지가 남녀 커플인 폴더

img_size = (224, 224)
batch_size = 8

# Load all image paths
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Assign labels (all images are labeled as 1 for male-female couple)
labels = [1] * len(image_files)

# Load and preprocess images
images = []
for img_path in image_files:
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    images.append(img_array)

# Convert to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Split into training and validation datasets
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# ----------------------------------------
# 2. Define a CNN model
# ----------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1.0 / 255, input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),  # Regularization
    tf.keras.layers.Dense(1, activation="sigmoid")  # Binary classification
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------------------
# 3. Train the CNN model
# ----------------------------------------
epochs = 10
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# Save the model
model_save_path = os.path.join(os.getcwd(), "couple_model.h5")
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# ----------------------------------------
# 4. Evaluate the model
# ----------------------------------------
print("\nEvaluating on validation data:")
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# ----------------------------------------
# 5. Plot Training & Validation Metrics
# ----------------------------------------
# Extract history data
training_acc = history.history['accuracy']
validation_acc = history.history['val_accuracy']
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range = range(epochs)

# Plot Accuracy
plt.figure(figsize=(12, 6))
plt.plot(epochs_range, training_acc, label='Training Accuracy')
plt.plot(epochs_range, validation_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss
plt.figure(figsize=(12, 6))
plt.plot(epochs_range, training_loss, label='Training Loss')
plt.plot(epochs_range, validation_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()