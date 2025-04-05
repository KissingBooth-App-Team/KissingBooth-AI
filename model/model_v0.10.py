import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

##############################################################################
# 0. 하이퍼파라미터 & 설정
##############################################################################
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 300  # 필요 시 64, 128 등으로 조정 가능
EPOCHS = 20
TEMPERATURE = 0.07  # 필요 시 0.07, 0.2 등으로 실험 가능
LEARNING_RATE = 1e-4

DATA_DIR = "/Users/dexterj/Desktop/model/merge"

##############################################################################
# 1. 데이터셋 준비
##############################################################################
all_image_paths = [
    os.path.join(DATA_DIR, fname)
    for fname in os.listdir(DATA_DIR)
    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
]
print(f"Total images found: {len(all_image_paths)}")

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0  # [0, 1] 범위로 정규화
    return image

@tf.function
def augment_image(image):
    # SimCLR에선 다양한 증강이 중요합니다
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    # 추가로 random crop, color jitter 등도 적용 가능
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def simclr_dataset(image_paths, batch_size):
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
    weights='imagenet',
    pooling='avg',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
# 사전학습된 백본도 학습되도록 설정
base_encoder.trainable = True

# Projection Head 예시: 중간 차원을 크게 두고, 최종 임베딩(128)
projection_head = models.Sequential([
    layers.Dense(2048, activation='relu'),  # 중간 차원을 늘려봄
    layers.Dense(128)                       # 최종 임베딩 차원
], name="projection_head")

##############################################################################
# 3. InfoNCE(Contrastive) Loss 정의
#   - 라벨 생성 부분을 보다 명시적으로 수정
##############################################################################
@tf.function
def info_nce_loss(features, temperature=TEMPERATURE):
    # features: [2N, embed_dim]
    # L2 정규화
    features = tf.math.l2_normalize(features, axis=1)

    # 유사도 행렬: [2N, 2N]
    similarity_matrix = tf.matmul(features, features, transpose_b=True)

    # 대각선(자기 자신) 제외용 mask
    batch_size_2n = tf.shape(features)[0]
    mask = tf.eye(batch_size_2n)

    # 스케일링
    logits = similarity_matrix / temperature
    # 자기 자신 위치에 -1e9로 마스킹
    logits_masked = tf.where(mask == 1, -1e9, logits)

    # (0,1), (2,3), (4,5), ... 이런 식으로 양성쌍을 매칭한다고 가정
    # => 짝수 인덱스의 positive는 다음(홀수 인덱스), 홀수 인덱스의 positive는 바로 이전(짝수 인덱스)
    labels = tf.concat([
        tf.range(1, batch_size_2n, 2),  # 짝수 -> 홀수
        tf.range(0, batch_size_2n, 2)   # 홀수 -> 짝수
    ], axis=0)
    # (주의) 위 로직은 배치 내부가 [img1_0, img2_0, img1_1, img2_1, ...] 순서로
    #       0,1 / 2,3 / 4,5 이렇게 들어온다는 가정하에 동작합니다.

    labels_one_hot = tf.one_hot(labels, depth=batch_size_2n)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_one_hot,
        logits=logits_masked
    )
    loss = tf.reduce_mean(loss)
    return loss

##############################################################################
# 4. 학습 루프 구성 (Custom Training Loop)
##############################################################################
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

@tf.function
def train_step(imgs1, imgs2):
    # imgs1, imgs2: [N, H, W, 3]
    with tf.GradientTape() as tape:
        # 2N개를 합쳐서 한 번에 base encoder -> projection head 통과
        concat_input = tf.concat([imgs1, imgs2], axis=0)  # [2N, H, W, 3]
        feats = base_encoder(concat_input, training=True)
        z = projection_head(feats)
        loss = info_nce_loss(z)

    # Gradient 계산
    grads = tape.gradient(loss, base_encoder.trainable_variables + projection_head.trainable_variables)
    # 적용
    optimizer.apply_gradients(zip(grads, base_encoder.trainable_variables + projection_head.trainable_variables))

    return loss

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for step, (img_batch1, img_batch2) in enumerate(train_ds):
        loss_value = train_step(img_batch1, img_batch2)
        if step % 10 == 0:
            print(f" Step {step}, InfoNCE Loss = {loss_value.numpy():.4f}")

##############################################################################
# 5. 학습 완료 후, 실제로 임베딩 추출 & 유사도 계산 + 저장
##############################################################################
# 5-1. 임베딩 모델 구성
embedding_model = tf.keras.Sequential([
    base_encoder,
    # projection_head  # 원한다면 포함 가능
])

# 5-2. 모델 저장
embedding_model.save("embedding_model_v0.10.h5")
print("Embedding model saved to embedding_model_v0.10.h5")

# 5-3. 전체 데이터에 대한 임베딩 추출
all_embeddings = []
for path in all_image_paths:
    img = load_and_preprocess_image(path)
    img = tf.expand_dims(img, axis=0)
    emb = embedding_model(img, training=False)
    emb = tf.math.l2_normalize(emb, axis=1)
    all_embeddings.append(emb.numpy())

all_embeddings = np.concatenate(all_embeddings, axis=0)
print("Embeddings shape:", all_embeddings.shape)

# 5-4. 임베딩 및 이미지 경로 저장 (추론 시 활용)
np.save("embedding_model_v0.10_embedding.npy", all_embeddings)
np.save("embedding_model_v0.10_image_path.npy", np.array(all_image_paths))

print("Saved embedding_model_v0.10_embedding.npy and embedding_model_v0.10_image_path.npy")
print("Done.")
