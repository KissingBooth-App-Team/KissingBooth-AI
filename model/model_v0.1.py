import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. Create the "couple" image datasets
# ----------------------------------------
data_dir = "/Users/dexterj/Desktop/cropping/merge"  # '커플' 이미지 폴더 경로

# Hyperparameters
batch_size = 8
img_size = (224, 224)

# Load image datasets
dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels=None,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    validation_split=0.2,
    subset='training',
    seed=42
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels=None,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    validation_split=0.2,
    subset='validation',
    seed=42
)

# ----------------------------------------
# 2. Add regression label (1.0) to all images
# ----------------------------------------
def add_regression_label(ds, label_value=1.0):
    """
    Add a regression label to the dataset.
    """
    return ds.map(lambda x: (x, tf.fill((tf.shape(x)[0], 1), label_value)))

train_dataset = add_regression_label(dataset, label_value=1.0)
val_dataset = add_regression_label(val_dataset, label_value=1.0)

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# ----------------------------------------
# 3. Define a simple CNN regression model
# ----------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Regression output
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mean_squared_error',
    metrics=['mean_squared_error', 'mean_absolute_error']  # Include MAE as a metric
)

# ----------------------------------------
# 4. Train the regression model
# ----------------------------------------
epochs = 5
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# Save the model to the current directory in h5 format
model_save_path = os.path.join(os.getcwd(), "model_v0.1.h5")
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# ----------------------------------------
# 5. Evaluate & Predict
# ----------------------------------------
print("\nEvaluating on validation data:")
val_loss, val_mse, val_mae = model.evaluate(val_dataset)  # Get MAE from evaluation
print(f"Validation MSE: {val_mse:.4f}")
print(f"Validation MAE (proxy for accuracy): {val_mae:.4f}")

# Predict values for a batch from validation dataset
print("\nPredicting values for validation data:")
for images, _ in val_dataset.take(1):
    preds = model.predict(images)
    for i, pred in enumerate(preds):
        print(f"Image {i} -> predicted value = {pred[0]:.4f}")

# ----------------------------------------
# 6. Plot Training & Validation Metrics
# ----------------------------------------
# Extract history data
training_mse = history.history['mean_squared_error']
validation_mse = history.history['val_mean_squared_error']
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs = range(len(training_mse))

# Plot MSE
plt.plot(epochs, training_mse, 'bo-', label='Training MSE')
plt.plot(epochs, validation_mse, 'b-', label='Validation MSE')
plt.title('Training & Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss
plt.figure()
plt.plot(epochs, training_loss, 'go-', label='Training Loss')
plt.plot(epochs, validation_loss, 'g-', label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
