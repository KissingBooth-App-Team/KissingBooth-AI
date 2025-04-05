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
TEMPERATURE = 0.03      
LEARNING_RATE = 1e-4

NORMAL_DIR = "/Users/dexterj/Desktop/model/merge"    
OUTLIER_DIR = "/Users/dexterj/Desktop/model/outlier"  

##############################################################################
# 1. 데이터 준비: 이미지 경로와 라벨 수집
##############################################################################
def get_image_paths(normal_dir, outlier_dir):
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
    return normal_paths, outlier_paths

normal_paths, outlier_paths = get_image_paths(NORMAL_DIR, OUTLIER_DIR)
print(f"Normal images: {len(normal_paths)}, Outlier images: {len(outlier_paths)}")

##############################################################################
# 2. 이미지 로드 및 증강 함수
##############################################################################
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

##############################################################################
# 3. Supervised Contrastive Loss 함수 (분모 0 처리 포함)
##############################################################################
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
    
    denom = tf.reduce_sum(mask, axis=1)
    denom = tf.where(tf.equal(denom, 0), tf.ones_like(denom), denom)
    
    mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / denom
    loss = -tf.reduce_mean(mean_log_prob_pos)
    return loss

##############################################################################
# 4. 클래스별 데이터셋 생성 및 배치 균형 유지
##############################################################################
def create_class_dataset(paths, label, batch_size_half):
    ds = tf.data.Dataset.from_tensor_slices((paths, [label] * len(paths)))
    ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = load_and_preprocess_image(path)
        img_aug1 = augment_image(img)
        img_aug2 = augment_image(img)
        return (img_aug1, img_aug2), (label, label)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size_half, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

half_batch = BATCH_SIZE // 2
normal_ds = create_class_dataset(normal_paths, 0, half_batch)
outlier_ds = create_class_dataset(outlier_paths, 1, half_batch)

balanced_ds = tf.data.Dataset.zip((normal_ds, outlier_ds))

def combine_batches(normal_batch, outlier_batch):
    (imgs1_n, imgs2_n), (labels1_n, labels2_n) = normal_batch
    (imgs1_o, imgs2_o), (labels1_o, labels2_o) = outlier_batch

    imgs1 = tf.concat([imgs1_n, imgs1_o], axis=0)
    imgs2 = tf.concat([imgs2_n, imgs2_o], axis=0)
    labels1 = tf.concat([labels1_n, labels1_o], axis=0)
    labels2 = tf.concat([labels2_n, labels2_o], axis=0)
    
    return (imgs1, imgs2), (labels1, labels2)

balanced_ds = balanced_ds.map(combine_batches, num_parallel_calls=tf.data.AUTOTUNE)

# 데이터셋의 예상 스텝 수 출력
steps_per_epoch = tf.data.experimental.cardinality(balanced_ds).numpy()
print(f"Estimated steps per epoch: {steps_per_epoch}")

##############################################################################
# 5. 모델 구성: Base Encoder와 Projection Head
##############################################################################
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

##############################################################################
# 6. 학습 단계 정의
##############################################################################
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

##############################################################################
# 7. 학습 루프 실행
##############################################################################
for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    step_count = 0
    for step, ((img_batch1, img_batch2), (label_batch1, label_batch2)) in enumerate(balanced_ds):
        loss_value = train_step((img_batch1, img_batch2), (label_batch1, label_batch2))
        step_count += 1
        if step % 10 == 0:
            print(f"  Step {step}, Loss = {loss_value.numpy():.4f}")
    print(f"Epoch {epoch} completed with {step_count} steps.")

##############################################################################
# 8. 학습 완료 후, 임베딩 모델 구성 & 저장
##############################################################################
embedding_model = tf.keras.Sequential([
    base_encoder,
    # projection_head 선택적 포함 가능
], name="embedding_model")

embedding_model.save("embedding_model_v2.5.h5")
print("Embedding model saved to embedding_model_v2.5.h5")

all_embeddings = []
for path in normal_paths + outlier_paths:
    img = load_and_preprocess_image(path)
    img = tf.expand_dims(img, axis=0)
    emb = embedding_model(img, training=False)
    emb = tf.math.l2_normalize(emb, axis=1)
    all_embeddings.append(emb.numpy())

all_embeddings = np.concatenate(all_embeddings, axis=0)
print("Embeddings shape:", all_embeddings.shape)

np.save("embedding_model_embedding_v2.5.npy", all_embeddings)
np.save("embedding_model_image_path_v2.5.npy", np.array(normal_paths + outlier_paths))
print("Saved embeddings and image paths.")
print("Done.")