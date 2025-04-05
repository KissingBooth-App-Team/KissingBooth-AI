import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

##############################################################################
# 0. 하이퍼파라미터 & 설정
##############################################################################
# 이미지 사이즈
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 200
EPOCHS = 5

# 데이터 디렉토리 (이미지 여러 장이 들어있는 폴더)
DATA_DIR = "/Users/dexterj/Desktop/model/merge"

##############################################################################
# 1. 데이터셋 준비
#    SimCLR에서는 배치 하나에 (N장의 원본 이미지 × 2개의 Augmentation) = 2N 이미지를
#    한꺼번에 forward 하며 InfoNCE Loss를 구합니다.
##############################################################################
# 1-1. 이미지 파일 목록 가져오기
all_image_paths = [
    os.path.join(DATA_DIR, fname)
    for fname in os.listdir(DATA_DIR)
    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
]

print(f"Total images found: {len(all_image_paths)}")

# 1-2. 이미지 로드 및 전처리 함수
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    # [0, 1] 스케일로 정규화
    image = image / 255.0
    return image

# 1-3. 데이터 증강 함수
@tf.function
def augment_image(image):
    # 예시로 넣은 간단한 augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

# 1-4. TF Dataset 구성
def simclr_dataset(image_paths, batch_size):
    """원본 이미지를 로드한 뒤, 서로 다른 두 번의 augmentation을 적용하여
       (augmented1, augmented2) 형태로 묶어 배치화하는 Dataset을 반환."""
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
# 2-1. Backbone (예시: ResNet50, ImageNet Pretrained)
base_encoder = tf.keras.applications.ResNet50(
    include_top=False, 
    weights='imagenet', 
    pooling='avg',   # GlobalAveragePooling
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# 2-2. Projection Head
projection_head = models.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(64)  # 출력 임베딩 차원 (예시)
], name="projection_head")

##############################################################################
# 3. InfoNCE(Contrastive) Loss 정의
##############################################################################
@tf.function
def info_nce_loss(features, temperature=0.1):
    """
    features: shape [2N, dim] (batch_size = N, 각 이미지당 2개 증강)
    temperature: softmax 온도 파라미터
    """
    # L2 정규화 (cosine similarity용)
    features = tf.math.l2_normalize(features, axis=1)

    # similarity matrix: (2N x 2N)
    similarity_matrix = tf.matmul(features, features, transpose_b=True)

    # (2N, 2N) 대각선 mask
    batch_size_2n = tf.shape(features)[0]
    mask = tf.eye(batch_size_2n)

    # similarity_matrix / temperature
    logits = similarity_matrix / temperature

    # 양성 쌍 index (2k, 2k+1) 등으로 구성
    labels = tf.range(batch_size_2n)
    labels_reshaped = tf.reshape(labels, (batch_size_2n // 2, 2))
    labels_swapped = tf.reverse(labels_reshaped, axis=[1])  # (0,1)->(1,0)
    labels_swapped = tf.reshape(labels_swapped, [-1])
    labels_one_hot = tf.one_hot(labels_swapped, depth=batch_size_2n)

    # 대각선(mask=1) 부분은 무시하기 위해 -1e9로 설정
    logits_masked = tf.where(mask == 1, -1e9, logits)

    # Cross Entropy 계산
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_one_hot,
        logits=logits_masked
    )
    loss = tf.reduce_mean(loss)
    return loss

##############################################################################
# 4. 학습 루프 구성 (Custom Training Loop)
##############################################################################
optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(imgs1, imgs2):
    # imgs1, imgs2: 서로 다른 augmentation을 거친 동일 원본의 배치 (shape: [N, H, W, 3])
    with tf.GradientTape() as tape:
        concat_input = tf.concat([imgs1, imgs2], axis=0)  # [2N, H, W, 3]
        feats = base_encoder(concat_input, training=True)
        z = projection_head(feats)
        loss = info_nce_loss(z)

    grads = tape.gradient(loss, base_encoder.trainable_variables + projection_head.trainable_variables)
    optimizer.apply_gradients(zip(grads, base_encoder.trainable_variables + projection_head.trainable_variables))
    
    return loss

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for step, (img_batch1, img_batch2) in enumerate(train_ds):
        loss_value = train_step(img_batch1, img_batch2)
        if step % 10 == 0:
            print(f" Step {step}, InfoNCE Loss = {loss_value.numpy():.4f}")

##############################################################################
# 5. 학습 완료 후, 실제로 임베딩 추출 & 유사도 계산
##############################################################################
# Embedding Model 구성: 보통 Backbone까지만 쓰는 경우가 많음
# (원하면 Projection Head까지 붙일 수도 있습니다)
embedding_model = tf.keras.Sequential([
    base_encoder,
    # projection_head  # <-- 원하면 추가
])

# 모델 저장(예: 전체 모델 구조를 h5로 저장)
# -------------------------------------------------
embedding_model.save("embedding_model_v0.5.h5")
print("Embedding model saved to embedding_model_v0.5.h5")

# 또는, base_encoder와 projection_head를 따로 저장 가능:
# base_encoder.save("simclr_base_encoder.h5")
# projection_head.save("simclr_projection_head.h5")

# 5-1. 전체 데이터에 대한 임베딩 추출
all_embeddings = []
for path in all_image_paths:
    # 이미지 로드 & 전처리
    img = load_and_preprocess_image(path)
    img = tf.expand_dims(img, axis=0)  # (1, H, W, 3)

    # 임베딩 추출
    emb = embedding_model(img, training=False)
    emb = tf.math.l2_normalize(emb, axis=1)  # cosine 유사도 계산 편의를 위해 normalize
    all_embeddings.append(emb.numpy())

all_embeddings = np.concatenate(all_embeddings, axis=0)  # (num_images, embedding_dim)
print("Embeddings shape:", all_embeddings.shape)

