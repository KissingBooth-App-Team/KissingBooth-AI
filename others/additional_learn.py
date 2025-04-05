#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SimCLR-like training + 이어서 학습(Continue Training) 예시 코드
==============================================================
1) Part A: 처음 학습 후 'embedding_model_v0.13.h5'로 저장
2) Part B: 저장된 모델 로드 후 projection head 다시 붙여서 추가 학습
3) 최종 모델 'embedding_model_v0.14.h5'로 저장
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

##############################################################################
# 공통 설정 (하이퍼파라미터 & 데이터셋 준비)
##############################################################################

IMG_HEIGHT = 224
IMG_WIDTH = 224

# GPU 메모리에 따라 조절하세요.
BATCH_SIZE = 400

# 첫 번째(초기) 학습할 Epoch 수
EPOCHS_INITIAL = 2

# 추가(이어) 학습할 Epoch 수
EPOCHS_CONTINUE = 3

TEMPERATURE = 0.07
LEARNING_RATE = 1e-4

# 데이터가 들어있는 디렉토리 (본인 환경에 맞춰 수정하세요)
DATA_DIR = "/Users/dexterj/Desktop/model/merge"

# (옵션) RandomCrop 설정
CROP_PROPORTION = 0.8
CROP_MIN = int(IMG_HEIGHT * CROP_PROPORTION)
CROP_MAX = IMG_HEIGHT

# -----------------------------------------------
# 파일 목록 수집
# -----------------------------------------------
all_image_paths = [
    os.path.join(DATA_DIR, fname)
    for fname in os.listdir(DATA_DIR)
    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
]
print(f"Total images found: {len(all_image_paths)}")


##############################################################################
# 데이터셋 구성 함수들
##############################################################################

def load_and_preprocess_image(path):
    """
    (1) 이미지 로드 -> (2) 224x224 리사이즈 -> (3) [0,1] 정규화
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0
    return image


@tf.function
def color_jitter(image, s=0.5):
    """
    밝기(brightness), 대비(contrast), 채도(saturation), hue 등을 임의 변경
    """
    image = tf.image.random_brightness(image, max_delta=0.8 * s)
    image = tf.image.random_contrast(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    image = tf.image.random_saturation(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    image = tf.image.random_hue(image, max_delta=0.2 * s)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

@tf.function
def augment_image(image):
    """
    SimCLR 스타일 증강: random crop, random flip, color jitter
    """
    image = tf.image.random_flip_left_right(image)
    image = color_jitter(image, s=0.5)
    return image

def simclr_dataset(image_paths, batch_size):
    """
    SimCLR: (aug1, aug2) 쌍을 동일 이미지에서 만들기
    """
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    path_ds = path_ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    def _map_fn(path):
        img = load_and_preprocess_image(path)
        img_aug1 = augment_image(img)
        img_aug2 = augment_image(img)
        return img_aug1, img_aug2

    ds = path_ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)  # drop_remainder=True 권장
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = simclr_dataset(all_image_paths, BATCH_SIZE)


##############################################################################
# InfoNCE Loss 정의
##############################################################################
@tf.function
def info_nce_loss(features, temperature=TEMPERATURE):
    """
    SimCLR InfoNCE Loss
    -------------------
    features: [2N, embed_dim] (aug1 + aug2 합친 것)
    """
    # L2 정규화
    features = tf.math.l2_normalize(features, axis=1)

    # 유사도 행렬: [2N, 2N]
    similarity_matrix = tf.matmul(features, features, transpose_b=True)

    # 자기 자신 제외용 마스킹
    batch_size_2n = tf.shape(features)[0]
    mask = tf.eye(batch_size_2n)

    # logits
    logits = similarity_matrix / temperature
    logits = tf.where(mask == 1, -1e9, logits)

    # 라벨 생성 (공식 SimCLR 방식)
    batch_size_n = batch_size_2n // 2
    labels = tf.concat([
        tf.range(batch_size_n, batch_size_2n),  # 앞쪽 N개 -> 뒤쪽 N개가 positive
        tf.range(0, batch_size_n)               # 뒤쪽 N개 -> 앞쪽 N개가 positive
    ], axis=0)

    labels_one_hot = tf.one_hot(labels, depth=batch_size_2n)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_one_hot,
        logits=logits
    )
    return tf.reduce_mean(loss)


##############################################################################
# Part A: SimCLR 초기 학습 (Base encoder + Projection head) & 저장
##############################################################################

print("\n==============================")
print(" Part A: SimCLR 초기 학습 시작 ")
print("==============================")

# --- 1) base_encoder (ResNet50) ---
base_encoder = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',  # ImageNet 사전학습
    pooling='avg',       # Global Average Pooling
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# (옵션) 일부 레이어만 학습 등 설정 가능
# for layer in base_encoder.layers[:-10]:
#     layer.trainable = False

# --- 2) projection_head ---
projection_head = models.Sequential([
    layers.Dense(2048, activation='relu'),
    layers.Dense(128)
], name="projection_head")

# --- 3) train_step 함수 정의 ---
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

@tf.function
def train_step(imgs1, imgs2):
    with tf.GradientTape() as tape:
        concat_input = tf.concat([imgs1, imgs2], axis=0)  # [2N, H, W, 3]
        feats = base_encoder(concat_input, training=True)
        z = projection_head(feats, training=True)
        loss = info_nce_loss(z, temperature=TEMPERATURE)
    vars_to_update = base_encoder.trainable_variables + projection_head.trainable_variables
    grads = tape.gradient(loss, vars_to_update)
    optimizer.apply_gradients(zip(grads, vars_to_update))
    return loss

# --- 4) 학습 루프 ---
for epoch in range(1, EPOCHS_INITIAL + 1):
    print(f"\nEpoch {epoch}/{EPOCHS_INITIAL} (Initial Training)")
    for step, (img_batch1, img_batch2) in enumerate(train_ds):
        loss_value = train_step(img_batch1, img_batch2)
        if step % 10 == 0:
            print(f"  Step {step}, InfoNCE Loss = {loss_value.numpy():.4f}")

# --- 5) 학습 완료 후, base_encoder만 저장할 수도 있고,
#         base_encoder + projection_head를 함께 저장할 수도 있음.
# 여기서는 "base_encoder만" 저장한다고 가정(본 예시와 동일)
embedding_model_initial = tf.keras.Sequential([
    base_encoder
], name="embedding_model_initial")

embedding_model_initial.save("embedding_model_v0.13.h5")
print("\n[Part A] Saved 'embedding_model_v0.13.h5' (base encoder only).")


##############################################################################
# Part B: 저장된 모델 로드 -> Projection head 재구성 -> 이어서 학습 -> 저장
##############################################################################

print("\n=========================================")
print(" Part B: 저장된 모델 로드 후 추가 학습 시작 ")
print("=========================================")

# --- 1) 로드 ---
loaded_model = tf.keras.models.load_model("embedding_model_v0.13.h5")
loaded_model.summary()
# 로드된 것은 base_encoder만 있을 것이라 가정
base_encoder_loaded = loaded_model.layers[0]
print("Base encoder loaded.")

# --- 2) projection_head 새로 정의 (다시 학습하려는 가정) ---
projection_head2 = models.Sequential([
    layers.Dense(2048, activation='relu'),
    layers.Dense(128)
], name="projection_head_2")

# --- 3) 이어 학습을 위한 info_nce_loss, optimizer, train_step 재정의 ---
optimizer2 = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

@tf.function
def train_step_continue(imgs1, imgs2):
    with tf.GradientTape() as tape:
        concat_input = tf.concat([imgs1, imgs2], axis=0)
        feats = base_encoder_loaded(concat_input, training=True)
        z = projection_head2(feats, training=True)
        loss = info_nce_loss(z, temperature=TEMPERATURE)
    vars_to_update = base_encoder_loaded.trainable_variables + projection_head2.trainable_variables
    grads = tape.gradient(loss, vars_to_update)
    optimizer2.apply_gradients(zip(grads, vars_to_update))
    return loss

# --- 4) 이어서 학습 루프 (추가 Epoch) ---
for epoch in range(1, EPOCHS_CONTINUE + 1):
    print(f"\nEpoch {epoch}/{EPOCHS_CONTINUE} (Continue Training)")
    for step, (img_batch1, img_batch2) in enumerate(train_ds):
        loss_value = train_step_continue(img_batch1, img_batch2)
        if step % 10 == 0:
            print(f"  Step {step}, InfoNCE Loss = {loss_value.numpy():.4f}")

# --- 5) 최종 모델( base_encoder + projection_head2 )로 저장 ---
embedding_model_final = tf.keras.Sequential([
    base_encoder_loaded,
    projection_head2
], name="embedding_model_final")

embedding_model_final.save("embedding_model_v0.14.h5")
print("\n[Part B] Saved 'embedding_model_v0.14.h5' (base encoder + new projection head).")


##############################################################################
# (선택) 모델 임베딩 뽑기 예시
##############################################################################

# 예: 최종 모델에서 base_encoder만 사용하여 임베딩 추출
#     혹은 projection_head까지 사용해도 됨(사용 목적에 따라).
print("\n추가: Embedding 추출 예시...")
embedding_only_model = tf.keras.Sequential([
    base_encoder_loaded
], name="embedding_only_model_after_continue")

all_embeddings = []
for path in all_image_paths:
    img = load_and_preprocess_image(path)
    img = tf.expand_dims(img, 0)  # [1, H, W, 3]
    emb = embedding_only_model(img, training=False)
    emb = tf.math.l2_normalize(emb, axis=1)
    all_embeddings.append(emb.numpy())

all_embeddings = np.concatenate(all_embeddings, axis=0)
print(f"Embeddings shape: {all_embeddings.shape}")

np.save("embedding_model_embedding_v0.14.npy", all_embeddings)
np.save("embedding_model_image_path_v0.14.npy", np.array(all_image_paths))
print("Saved embedding_model_embedding_v0.14.npy and image_paths_v0.14.npy")

print("\n=== Done. ===")
