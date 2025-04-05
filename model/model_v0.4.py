import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. Load images and assign labels
# ----------------------------------------
data_dir = "/Users/dexterj/Desktop/cropping/merge"  # 남녀 커플 이미지 폴더

img_size = (224, 224)
batch_size = 8

# Load all image paths
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Assign labels (1 for all images in the current dataset)
labels = [1] * len(image_files)  # 모든 이미지를 "남녀 커플"로 레이블링

# Load and preprocess images
images = []
for img_path in image_files:
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    images.append(img_array)

images = np.array(images)
labels = np.array(labels)

# Split data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# ----------------------------------------
# 2. Define a feature extraction model
# ----------------------------------------
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------------------
# 3. Train the model
# ----------------------------------------
epochs = 10

history = model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=epochs,
    batch_size=batch_size
)

# ----------------------------------------
# 4. Save the model
# ----------------------------------------
model_save_path = os.path.join(os.getcwd(), "model_v0.4.h5")
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# ----------------------------------------
# 5. Evaluate the model
# ----------------------------------------
val_loss, val_accuracy = model.evaluate(val_images, val_labels)
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# ----------------------------------------
# 6. Plot Training & Validation Metrics
# ----------------------------------------
training_acc = history.history['accuracy']
validation_acc = history.history['val_accuracy']
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.plot(epochs_range, training_acc, label='Training Accuracy')
plt.plot(epochs_range, validation_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(epochs_range, training_loss, label='Training Loss')
plt.plot(epochs_range, validation_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------
# 7. Load the saved model (Optional)
# ----------------------------------------
loaded_model = tf.keras.models.load_model(model_save_path)
print("Model loaded successfully.")