import tensorflow as tf
import os

# ----------------------------------------
# 1. Load the existing model
# ----------------------------------------
model_load_path = os.path.join(os.getcwd(), "model_v0.4.h5")  # Path to the saved model in h5 format
model = tf.keras.models.load_model(model_load_path)
print(f"Model loaded from {model_load_path}")

# ----------------------------------------
# 2. Prepare the test dataset
# ----------------------------------------
data_dir = "/Users/dexterj/Desktop/model/testdata"  # Same directory as training/validation data

# Hyperparameters
batch_size = 8
img_size = (224, 224)

# Load the test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels=None,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=False  # Avoid shuffling for evaluation
)

# Add regression labels to the test dataset
def add_regression_label(ds, label_value=1.0):
    return ds.map(lambda x: (x, tf.fill((tf.shape(x)[0], 1), label_value)))

test_dataset = add_regression_label(test_dataset, label_value=1.0)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# ----------------------------------------
# 3. Evaluate the model on the test dataset
# ----------------------------------------
print("\nEvaluating on test data:")
test_loss, test_accuracy = model.evaluate(test_dataset)  # Adjusted unpacking
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ----------------------------------------
# 4. Predict values for a batch from the test dataset
# ----------------------------------------
print("\nPredicting values for test data:")
for images, _ in test_dataset.take(1):
    preds = model.predict(images)
    for i, pred in enumerate(preds):
        print(f"Test Image {i} -> predicted value = {pred[0]:.4f}")