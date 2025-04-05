#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SimCLR-like training example with:
  - Improved label creation in InfoNCE
  - Additional data augmentations (random crop, color jitter)
  - Optionally partially-freezing the base encoder
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

##############################################################################
# 0. 하이퍼파라미터 & 설정
##############################################################################
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 300     # GPU 상황에 맞춰 조절
EPOCHS = 5         # 예시로 10 epoch (적절히 조정)
TEMPERATURE = 0.07  # SimCLR에서 흔히 사용하는 값
LEARNING_RATE = 1e-4

# 데이터가 들어있는 디렉토리
DATA_DIR = "/Users/dexterj/Desktop/model/merge"  # 각자 환경에 맞춰 수정하세요.

# (옵션) RandomCrop 설정
CROP_PROPORTION = 0.8    # 예: 80% 영역만 랜덤 크롭
CROP_MIN = int(IMG_HEIGHT * CROP_PROPORTION)
CROP_MAX = IMG_HEIGHT    # 224

##############################################################################
# 1. 데이터셋 준비 (파일 목록 & 증강 함수)
##############################################################################
all_image_paths = [
    os.path.join(DATA_DIR, fname)
    for fname in os.listdir(DATA_DIR)
    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
]
print(f"Total images found: {len(all_image_paths)}")

def load_and_preprocess_image(path):
    """
    기본 로드: [IMG_HEIGHT, IMG_WIDTH] 크기로 리사이즈 & [0,1]로 정규화
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    # 전체 크기로 리사이즈 후, 이후 augment에서 crop
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0
    return image

@tf.function
def random_crop(image):
    """
    Random cropping (ex: [CROP_MIN, CROP_MIN]) -> 다시 [IMG_HEIGHT, IMG_WIDTH]로 리사이즈
    """
    # 일단 crop_size를 랜덤 결정 (CROP_MIN ~ CROP_MAX)
    crop_size = tf.random.uniform(
        shape=[],
        minval=CROP_MIN,
        maxval=CROP_MAX,
        dtype=tf.int32
    )
    cropped = tf.image.random_crop(
        image, size=[crop_size, crop_size, 3]
    )
    # crop 이후 다시 224x224
    cropped = tf.image.resize(cropped, [IMG_HEIGHT, IMG_WIDTH])
    return cropped

@tf.function
def color_jitter(image, s=0.5):
    """
    Color jitter: brightness, contrast, saturation, hue 변경
    s 값으로 변동 강도 조절
    """
    # random brightness
    image = tf.image.random_brightness(image, max_delta=0.8 * s)
    # random contrast
    image = tf.image.random_contrast(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    # random saturation
    image = tf.image.random_saturation(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    # random hue
    image = tf.image.random_hue(image, max_delta=0.2 * s)
    # clip
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

@tf.function
def augment_image(image):
    """
    SimCLR에서 중요한: random crop, random flip, color jitter 등
    """
    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)
    image = color_jitter(image, s=0.5)
    return image

def simclr_dataset(image_paths, batch_size):
    """
    두 뷰(aug1, aug2)를 동시에 뽑아서 (aug1, aug2) 쌍으로 내보냄
    """
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    path_ds = path_ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    def _map_fn(path):
        img = load_and_preprocess_image(path)
        img_aug1 = augment_image(img)
        img_aug2 = augment_image(img)
        return img_aug1, img_aug2

    ds = path_ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = simclr_dataset(all_image_paths, BATCH_SIZE)

##############################################################################
# 2. SimCLR 모델 구성
##############################################################################
base_encoder = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',  # 사전학습된 Weight 사용
    pooling='avg',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# (옵션) 초반 몇 Epoch 동안 ResNet 일부(또는 전체) 동결할 수도 있음.
# 아래는 예시로 맨 마지막 Conv Block만 학습(Partial Fine-tuning)하도록 하는 방법:
# for layer in base_encoder.layers[:-10]:
#     layer.trainable = False

# Projection Head (간단한 예시)
projection_head = models.Sequential([
    layers.Dense(2048, activation='relu'),
    layers.Dense(128)
], name="projection_head")

##############################################################################
# 3. InfoNCE(Contrastive) Loss 정의
#    - 라벨 생성 부분 수정 (공식 SimCLR에서 흔히 쓰는 방식)
##############################################################################
@tf.function
def info_nce_loss(features, temperature=TEMPERATURE):
    """
    Args:
      features: [2N, embed_dim], concat된 z1+z2
    Returns:
      scalar loss
    """
    # L2 정규화
    features = tf.math.l2_normalize(features, axis=1)

    # 유사도 행렬: [2N, 2N]
    similarity_matrix = tf.matmul(features, features, transpose_b=True)

    # 자기 자신 제외용 mask
    batch_size_2n = tf.shape(features)[0]
    mask = tf.eye(batch_size_2n)

    # 유사도 -> logits
    logits = similarity_matrix / temperature
    logits = tf.where(mask == 1, -1e9, logits)

    # 라벨 생성 (공식 SimCLR 방식):
    # 앞쪽 N개 (0~N-1)은 뒤쪽 N개( N~2N-1 )가 positive
    # 뒤쪽 N개는 앞쪽 N개가 positive
    # => labels = [N, N+1, ..., 2N-1, 0, 1, ..., N-1]
    batch_size_n = batch_size_2n // 2
    labels = tf.concat([
        tf.range(batch_size_n, batch_size_2n),  # 0->N, 1->N+1, ...
        tf.range(0, batch_size_n)               # N->0, N+1->1, ...
    ], axis=0)

    labels_one_hot = tf.one_hot(labels, depth=batch_size_2n)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_one_hot,
        logits=logits
    )
    return tf.reduce_mean(loss)

##############################################################################
# 4. 학습 루프 구성 (Custom Training Loop)
##############################################################################
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

@tf.function
def train_step(imgs1, imgs2):
    """
    imgs1, imgs2: [N, H, W, 3]
    """
    with tf.GradientTape() as tape:
        # 2N개 합쳐서 한 번에 feature 추출
        concat_input = tf.concat([imgs1, imgs2], axis=0)  # [2N, H, W, 3]
        feats = base_encoder(concat_input, training=True)
        z = projection_head(feats, training=True)
        loss = info_nce_loss(z, temperature=TEMPERATURE)

    vars_to_update = base_encoder.trainable_variables + projection_head.trainable_variables
    grads = tape.gradient(loss, vars_to_update)
    optimizer.apply_gradients(zip(grads, vars_to_update))

    return loss

##############################################################################
# 5. 학습 실행
##############################################################################
for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    for step, (img_batch1, img_batch2) in enumerate(train_ds):
        loss_value = train_step(img_batch1, img_batch2)

        if step % 10 == 0:
            print(f"  Step {step}, InfoNCE Loss = {loss_value.numpy():.4f}")

##############################################################################
# 6. 학습 완료 후, 임베딩 모델 구성 & 저장
##############################################################################
embedding_model = tf.keras.Sequential([
    base_encoder,
    # projection_head  # (원하면 projection head까지 포함 가능)
], name="embedding_model")

embedding_model.save("embedding_model_v0.13.h5")
print("Embedding model saved to embedding_model_v0.13.h5")

# 임베딩 추출 (원하는 경우)
all_embeddings = []
for path in all_image_paths:
    img = load_and_preprocess_image(path)
    img = tf.expand_dims(img, axis=0)
    emb = embedding_model(img, training=False)
    emb = tf.math.l2_normalize(emb, axis=1)
    all_embeddings.append(emb.numpy())

all_embeddings = np.concatenate(all_embeddings, axis=0)
print("Embeddings shape:", all_embeddings.shape)

# 임베딩 및 이미지 경로 저장
np.save("embedding_model_embedding_v0.13.npy", all_embeddings)
np.save("embedding_model_image_path_v0.13.npy", np.array(all_image_paths))
print("Saved embedding_simclr.npy and image_paths_simclr.npy")
print("Done.")
