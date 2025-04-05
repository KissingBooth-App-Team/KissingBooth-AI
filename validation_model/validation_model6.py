import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

##############################################################################
# 1. 예측 시그니처
#    - 예측값(prediction)이 0.5보다 작으면 outlier(0), 크면 normal(1)로 분류
##############################################################################

IMG_HEIGHT = 224
IMG_WIDTH = 224

MODEL_PATH = "/Users/dexterj/Desktop/model/model_final.h5"  

TEST_IMAGE_PATH = "/Users/dexterj/Desktop/model/testdata/couple.jpg"

try:
    embedding_model = tf.keras.models.load_model(MODEL_PATH)
    print(f"임베딩 모델이 '{MODEL_PATH}' 에서 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    exit(1)

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0
    return image



sample_img = load_and_preprocess_image(TEST_IMAGE_PATH)
sample_img = tf.expand_dims(sample_img, axis=0)  # 배치 차원
prob = embedding_model.predict(sample_img)[0][0]          # 0~1 사이 확률
pred_label = 1 if prob > 0.5 else 0
print("Predicted label =", pred_label, "(0=outlier, 1=normal), Prob =", prob)