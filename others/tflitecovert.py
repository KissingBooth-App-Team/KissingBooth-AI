import tensorflow as tf

embedding_model = tf.keras.models.load_model("model_v0.4.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(embedding_model)
tflite_model = converter.convert()
with open("model_v0.4.tflite", "wb") as f:
    f.write(tflite_model)