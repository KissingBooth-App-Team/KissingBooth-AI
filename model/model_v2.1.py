#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

##############################################################################
# 0. 하이퍼파라미터 & 설정
##############################################################################
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 500     
EPOCHS = 5         
LEARNING_RATE = 1e-4

# 데이터 디렉토리 설정
NORMAL_DIR = "/Users/dexterj/Desktop/model/merge"    # 정상 이미지 폴더
OUTLIER_DIR = "/Users/dexterj/Desktop/model/outlier"  # 이상치 이미지 폴더

##############################################################################
# 1. 데이터 준비: 이미지 경로와 라벨 수집
##############################################################################
def get_image_paths_and_labels(normal_dir, outlier_dir):
    normal_paths = [
        os.path.join(normal_dir, fname) 
        for fname in os.listdir(normal_dir) 
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    outlier_paths = [
        os.path.join(outlier_dir, fname) 
        for fname in os.listdir(outlier_dir) 
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    image_paths = normal_paths + outlier_paths
    labels = [0] * len(normal_paths) + [1] * len(outlier_paths)
    
    return image_paths, labels

all_image_paths, all_labels = get_image_paths_and_labels(NORMAL_DIR, OUTLIER_DIR)
print(f"Total images: {len(all_image_paths)}, Normal: {all_labels.count(0)}, Outlier: {all_labels.count(1)}")

##############################################################################
# 2. 이미지 로드 및 전처리 함수
##############################################################################
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0
    return image

##############################################################################
# 3. 데이터셋 구성: 이진 분류용 데이터셋 생성
##############################################################################
def simple_dataset(image_paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = load_and_preprocess_image(path)
        return img, label

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = simple_dataset(all_image_paths, all_labels, BATCH_SIZE)

##############################################################################
# 4. 모델 구성: Base Encoder와 분류 헤드
##############################################################################
base_encoder = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    pooling='avg',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# 필요시 base_encoder 동결 가능
# for layer in base_encoder.layers:
#     layer.trainable = False

classifier_head = models.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')  # 2개 클래스: 정상(0), 이상치(1)
], name="classifier_head")

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_encoder(inputs, training=True)
outputs = classifier_head(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

##############################################################################
# 5. 모델 학습
##############################################################################
model.fit(train_ds, epochs=EPOCHS)

##############################################################################
# 6. 모델 저장
##############################################################################
model.save("embedding_model_v2.1.h5")
print("Model saved as binary_classification_model.h5")