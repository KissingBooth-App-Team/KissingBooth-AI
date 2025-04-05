#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 0. 하이퍼파라미터 & 설정
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 300     
EPOCHS = 20                # 에포크 수 증가
TEMPERATURE = 0.01         # 온도 낮춤
LEARNING_RATE = 1e-4

NORMAL_DIR = "/Users/dexterj/Desktop/model/merge"    
OUTLIER_DIR = "/Users/dexterj/Desktop/model/outlier"  

# 1. 데이터 준비
def get_image_paths_and_labels(normal_dir, outlier_dir):
    normal_paths = [os.path.join(normal_dir, fname) for fname in os.listdir(normal_dir) 
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
    outlier_paths = [os.path.join(outlier_dir, fname) for fname in os.listdir(outlier_dir) 
                     if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = normal_paths + outlier_paths
    labels = [0] * len(normal_paths) + [1] * len(outlier_paths)
    return image_paths, labels

all_image_paths, all_labels = get_image_paths_and_labels(NORMAL_DIR, OUTLIER_DIR)
print(f"Total images: {len(all_image_paths)}, Normal: {all_labels.count(0)}, Outlier: {all_labels.count(1)}")

# 2. 이미지 로드 및 증강
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0
    return image

@tf.function
def color_jitter(image, s=0.5):
    image = tf.image.random_brightness(image, max_delta=0.8 * s)
    image = tf.image.random_contrast(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    image = tf.image.random_saturation(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    image = tf.image.random_hue(image, max_delta=0.2 * s)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

@tf.function
def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = color_jitter(image, s=0.5)
    return image

# 3. Supervised Contrastive Loss
@tf.function
def supervised_contrastive_loss(features, labels, temperature=TEMPERATURE):
    features = tf.math.l2_normalize(features, axis=1)
    similarity_matrix = tf.matmul(features, features, transpose_b=True) / temperature

    batch_size = tf.shape(features)[0]
    logits_mask = tf.cast(tf.logical_not(tf.eye(batch_size, dtype=tf.bool)), tf.float32)

    labels = tf.expand_dims(labels, 1)
    label_mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)

    mask = logits_mask * label_mask
    exp_sim = tf.exp(similarity_matrix) * logits_mask

    log_prob = similarity_matrix - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-9)

    denom = tf.reduce_sum(mask, axis=1) + 1e-9
    mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / denom

    loss = -tf.reduce_mean(mean_log_prob_pos)
    return loss

# 4. 데이터셋 구성
def supervised_simclr_dataset(image_paths, labels, batch_size):
    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    path_ds = path_ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = load_and_preprocess_image(path)
        img_aug1 = augment_image(img)
        img_aug2 = augment_image(img)
        return (img_aug1, img_aug2), (label, label)

    ds = path_ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = supervised_simclr_dataset(all_image_paths, all_labels, BATCH_SIZE)

# 5. 모델 구성
base_encoder = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    pooling='avg',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
projection_head = models.Sequential([
    layers.Dense(2048, activation='relu'),
    layers.Dense(128)
], name="projection_head")

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# 6. 학습 단계
@tf.function
def train_step(batch_images, batch_labels):
    (imgs1, imgs2) = batch_images
    (labels1, labels2) = batch_labels

    concat_images = tf.concat([imgs1, imgs2], axis=0)
    concat_labels = tf.concat([labels1, labels2], axis=0)

    with tf.GradientTape() as tape:
        feats = base_encoder(concat_images, training=True)
        z = projection_head(feats, training=True)
        loss = supervised_contrastive_loss(z, concat_labels, temperature=TEMPERATURE)

    vars_to_update = base_encoder.trainable_variables + projection_head.trainable_variables
    grads = tape.gradient(loss, vars_to_update)
    optimizer.apply_gradients(zip(grads, vars_to_update))
    return loss

# 7. 콜백 설정
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1)
]

# 8. 학습 루프 실행
for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    for step, ((img_batch1, img_batch2), (label_batch1, label_batch2)) in enumerate(train_ds):
        loss_value = train_step((img_batch1, img_batch2), (label_batch1, label_batch2))
        if step % 10 == 0:
            print(f"  Step {step}, Loss = {loss_value.numpy():.4f}")

# 9. 학습 완료 후, 모델 저장
embedding_model = tf.keras.Sequential([
    base_encoder,
    # projection_head 선택적 포함 가능
], name="embedding_model")

embedding_model.save("embedding_model_v2.7.h5")
print("Embedding model saved to embedding_model_v2.7.h5")

# 임베딩 추출 (선택적)
all_embeddings = []
for path in all_image_paths:
    img = load_and_preprocess_image(path)
    img = tf.expand_dims(img, axis=0)
    emb = embedding_model(img, training=False)
    emb = tf.math.l2_normalize(emb, axis=1)
    all_embeddings.append(emb.numpy())

all_embeddings = np.concatenate(all_embeddings, axis=0)
print("Embeddings shape:", all_embeddings.shape)

np.save("embedding_model_embedding_v2.7.npy", all_embeddings)
np.save("embedding_model_image_path_v2.7.npy", np.array(all_image_paths))
print("Saved embeddings and image paths.")
print("Done.")