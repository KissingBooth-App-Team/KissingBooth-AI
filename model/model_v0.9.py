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
BATCH_SIZE = 200
EPOCHS = 5

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
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
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

projection_head = models.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(64)  # 출력 임베딩 차원
], name="projection_head")

##############################################################################
# 3. InfoNCE(Contrastive) Loss 정의
##############################################################################
@tf.function
def info_nce_loss(features, temperature=0.1):
    features = tf.math.l2_normalize(features, axis=1)
    similarity_matrix = tf.matmul(features, features, transpose_b=True)

    batch_size_2n = tf.shape(features)[0]
    mask = tf.eye(batch_size_2n)

    logits = similarity_matrix / temperature

    labels = tf.range(batch_size_2n)
    labels_reshaped = tf.reshape(labels, (batch_size_2n // 2, 2))
    labels_swapped = tf.reverse(labels_reshaped, axis=[1])
    labels_swapped = tf.reshape(labels_swapped, [-1])
    labels_one_hot = tf.one_hot(labels_swapped, depth=batch_size_2n)

    logits_masked = tf.where(mask == 1, -1e9, logits)
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
# 5. 학습 완료 후, 실제로 임베딩 추출 & 유사도 계산 + 저장
##############################################################################
# 5-1. 임베딩 모델 구성
embedding_model = tf.keras.Sequential([
    base_encoder,
    # projection_head  # (원하면 포함)
])

# 5-2. 모델 저장
embedding_model.save("embedding_model_v0.6.h5")
print("Embedding model saved to embedding_model_v0.6.h5")

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
np.save("embedding_model_v0.6_embedding.npy", all_embeddings)
np.save("embedding_model_v0.6_image_path.npy", np.array(all_image_paths))

print("Saved all_embeddings.npy and all_image_paths.npy")
print("Done.")