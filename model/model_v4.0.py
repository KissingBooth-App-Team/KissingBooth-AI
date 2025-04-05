import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

##############################################################################
# 0. 하이퍼파라미터 & 설정
##############################################################################
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 10      # 초기 학습 에포크 (헤드만 학습)
EPOCHS_STAGE2 = 10      # 미세 조정 에포크
LEARNING_RATE_STAGE1 = 1e-4
LEARNING_RATE_STAGE2 = 1e-5

NORMAL_DIR = "/Users/dexterj/Desktop/model/merge"
OUTLIER_DIR = "/Users/dexterj/Desktop/model/outlier"

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
    image_paths = outlier_paths + normal_paths
    labels = [0] * len(outlier_paths) + [1] * len(normal_paths)
    return image_paths, labels

all_image_paths, all_labels = get_image_paths_and_labels(NORMAL_DIR, OUTLIER_DIR)
print(f"Total images: {len(all_image_paths)}")
print(f"  Outlier(0): {all_labels.count(0)}, Normal(1): {all_labels.count(1)}")

##############################################################################
# 2. 이미지 로드 및 증강 함수 개선
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
    # 수평 뒤집기, 회전, 확대/축소 등의 추가 증강 적용
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = color_jitter(image, s=0.5)
    # 0~1 범위로 클리핑
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

##############################################################################
# 3. 데이터셋 구성
##############################################################################
def create_dataset(image_paths, labels, batch_size, shuffle=True):
    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        path_ds = path_ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = load_and_preprocess_image(path)
        img = augment_image(img)
        return img, label

    ds = path_ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(all_image_paths, all_labels, BATCH_SIZE, shuffle=True)

##############################################################################
# 4. 모델 구성: Base Encoder + 개량된 분류 헤드
##############################################################################
# 사전학습된 ResNet50 불러오기 (분류헤드 제외)
base_encoder = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    pooling='avg',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# 분류 헤드 개선: Dense 층 추가, Dropout 적용
def build_classification_head():
    return models.Sequential([
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name="classification_head")

classification_head = build_classification_head()

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_encoder(inputs, training=False)  # 초기에는 base_encoder 학습하지 않음
outputs = classification_head(x)
model = tf.keras.Model(inputs, outputs, name="OutlierVsNormal")

##############################################################################
# 5. 불균형 데이터 대비 (class_weight) 설정
##############################################################################
num_outlier = all_labels.count(0)
num_normal = all_labels.count(1)
total = num_outlier + num_normal

w_outlier = total / (2.0 * num_outlier + 1e-8)
w_normal = total / (2.0 * num_normal + 1e-8)
class_weight = {0: w_outlier, 1: w_normal}
print("Class Weights:", class_weight)

##############################################################################
# 6. 학습 준비: 콜백 설정
##############################################################################
early_stopping = callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1)
checkpoint = callbacks.ModelCheckpoint("best_model.h5", monitor='loss', save_best_only=True)

##############################################################################
# 7. Stage 1: 분류 헤드만 학습
##############################################################################
# base_encoder 고정
base_encoder.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_STAGE1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_stage1 = model.fit(
    train_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weight,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

##############################################################################
# 8. Stage 2: 미세 조정 (Fine-tuning)
##############################################################################
# base_encoder 일부 레이어 풀기 (예: 마지막 50개 레이어)
base_encoder.trainable = True
for layer in base_encoder.layers[:-50]:
    layer.trainable = False

# 학습률 낮춤
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_STAGE2),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_stage2 = model.fit(
    train_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weight,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

##############################################################################
# 9. 최종 모델 저장
##############################################################################
model.save("model_v4.0")
print("Improved model saved to model_v4.0.h5")