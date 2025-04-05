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
BATCH_SIZE = 300     
EPOCHS = 5                
LEARNING_RATE = 1e-4

# 데이터 디렉토리 설정
NORMAL_DIR = "/Users/dexterj/Desktop/model/merge"    
OUTLIER_DIR = "/Users/dexterj/Desktop/model/outlier"  

##############################################################################
# 1. 데이터 준비: 이미지 경로와 레이블 수집 및 라벨 변환
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
    # 기존 레이블: 정상 0, 이상치 1 -> 새로운 레이블: 정상 1, 이상치 0
    labels = [1] * len(normal_paths) + [0] * len(outlier_paths)
    image_paths = normal_paths + outlier_paths
    return image_paths, labels

all_image_paths, all_labels = get_image_paths_and_labels(NORMAL_DIR, OUTLIER_DIR)
print(f"Total images: {len(all_image_paths)}, Normal: {all_labels.count(1)}, Outlier: {all_labels.count(0)}")

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

classifier_head = models.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # 단일 뉴런, 정상 확률 출력
], name="classifier_head")

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_encoder(inputs, training=True)
outputs = classifier_head(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

##############################################################################
# 5. 콜백 및 클래스 가중치 설정 (선택 사항)
##############################################################################
class_weight = {0: 1.0, 1: 2.0}  # 이상치(0)보다는 정상(1)에 더 높은 가중치를 줄 수 있음

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1)
]

##############################################################################
# 6. 모델 학습
##############################################################################
model.fit(train_ds, epochs=EPOCHS, class_weight=class_weight, callbacks=callbacks)

##############################################################################
# 7. 모델 저장
##############################################################################
model.save("embedding_model_v2.4.h5")
print("Model saved as embedding_model_v2.4.h5")