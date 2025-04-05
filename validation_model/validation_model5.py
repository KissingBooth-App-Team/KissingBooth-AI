#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf

IMG_HEIGHT = 224
IMG_WIDTH = 224

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0
    return image

def main(image_path):
    # 학습된 모델 로드
    model = tf.keras.models.load_model("embedding_model_v2.1.h5")
    
    # 이미지 로드 및 전처리
    img = load_and_preprocess_image(image_path)
    img = tf.expand_dims(img, axis=0)  # 배치 차원 추가

    # 예측 수행
    predictions = model.predict(img)
    # predictions shape: (1, 2), 각 클래스에 대한 확률
    prob_normal = predictions[0][0]  # 정상(클래스 0) 확률
    prob_outlier = predictions[0][1] # 이상치(클래스 1) 확률

    print(f"Image: {image_path}")
    print(f"Probability Normal: {prob_normal*100:.2f}%, Probability Outlier: {prob_outlier*100:.2f}%")

    # 클래스 별로 결과 해석
    if prob_normal > prob_outlier:
        print("Predicted: Normal")
    else:
        print("Predicted: Outlier")

if __name__ == "__main__":
    image_path = "/Users/dexterj/Desktop/model/testdata/dogs.jpg"
    main(image_path)