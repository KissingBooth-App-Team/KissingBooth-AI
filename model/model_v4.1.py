import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

##############################################################################
# 0. 하이퍼파라미터 & 설정
##############################################################################
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

EPOCHS_STAGE1 = 10  # 초기 학습 에포크 (헤드만 학습)
EPOCHS_STAGE2 = 10  # 미세 조정 에포크

LEARNING_RATE_STAGE1 = 1e-3  # 헤드만 먼저 학습할 때는 약간 크게
LEARNING_RATE_STAGE2 = 1e-5  # 미세조정 시 조금 더 작게

NORMAL_DIR = "/Users/dexterj/Desktop/model/merge"
OUTLIER_DIR = "/Users/dexterj/Desktop/model/outlier"

##############################################################################
# 1. 데이터 준비: 이미지 경로와 라벨 수집
##############################################################################
def get_image_paths_and_labels(normal_dir, outlier_dir):
    # Normal=1, Outlier=0 라벨
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
# (선택) 오버피팅 테스트를 위한 소량 데이터 추출 코드
# - 이 부분은 필요 시 사용하세요.
##############################################################################
"""
# 예: 전체 데이터 중 20장만 샘플 추출
sample_count = 20
all_image_paths = all_image_paths[:sample_count]
all_labels = all_labels[:sample_count]
print(f"[오버피팅 테스트] {sample_count} 장만 사용합니다.")
"""

##############################################################################
# 2. Train/Validation 분리
##############################################################################
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths, all_labels, 
    test_size=0.2,      # 예: 20%를 검증용
    random_state=42, 
    stratify=all_labels # 불균형 시 라벨 비율 유지
)

print(f"Train set: {len(train_paths)}, Validation set: {len(val_paths)}")

##############################################################################
# 3. 이미지 로드 함수 & 증강 함수 (train/val 분리 적용)
##############################################################################
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    # 0~1 범위로 스케일
    image = image / 255.0
    return image

@tf.function
def color_jitter(image, s=0.2):
    # s가 작을수록 색 왜곡 정도가 감소
    image = tf.image.random_brightness(image, max_delta=0.8 * s)
    image = tf.image.random_contrast(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    image = tf.image.random_saturation(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    image = tf.image.random_hue(image, max_delta=0.2 * s)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

@tf.function
def augment_image_train(image):
    # 증강: 수평 뒤집기 + 색상 조정(강도 낮춤)
    image = tf.image.random_flip_left_right(image)
    image = color_jitter(image, s=0.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def create_dataset(paths, labels, batch_size, shuffle=True, augment=False):
    path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        path_ds = path_ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = load_and_preprocess_image(path)
        if augment:
            img = augment_image_train(img)
        return img, label

    ds = path_ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# 학습 세트는 증강 적용, 검증 세트는 증강 없음
train_ds = create_dataset(train_paths, train_labels, BATCH_SIZE, shuffle=True, augment=True)
val_ds   = create_dataset(val_paths,   val_labels,   BATCH_SIZE, shuffle=False, augment=False)

##############################################################################
# 4. 모델 구성: 사전학습된 ResNet50 + 커스텀 분류 헤드
##############################################################################
# 사전학습된 ResNet50 불러오기 (분류헤드 제외)
base_encoder = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    pooling='avg',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

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
x = base_encoder(inputs, training=False)
outputs = classification_head(x)
model = tf.keras.Model(inputs, outputs, name="OutlierVsNormal")

##############################################################################
# 5. (우선) 클래스 불균형 가중치 제거
#    - 문제 없이 학습 확인 후, 필요하면 class_weight 추가 또는 다른 방법 모색
##############################################################################
# (필요 시 재활성화 예시)
# num_outlier = all_labels.count(0)
# num_normal = all_labels.count(1)
# total = num_outlier + num_normal
# w_outlier = total / (2.0 * num_outlier + 1e-8)
# w_normal = total / (2.0 * num_normal + 1e-8)
# class_weight = {0: w_outlier, 1: w_normal}
# print("Class Weights:", class_weight)

##############################################################################
# 6. 콜백 설정 (val_loss 모니터링)
##############################################################################
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    "best_model.h5", 
    monitor='val_loss', 
    save_best_only=True
)

##############################################################################
# 7. Stage 1: 분류 헤드만 학습
##############################################################################
base_encoder.trainable = False  # 기존 특징 추출부는 동결
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_STAGE1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    # class_weight=class_weight,  # 일단 주석 처리
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

##############################################################################
# 8. Stage 2: 미세 조정 (Fine-tuning)
##############################################################################
# base_encoder의 일부 레이어만 풀어서 학습
base_encoder.trainable = True

# 예시: 끝에서부터 50개 레이어만 학습
for layer in base_encoder.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_STAGE2),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    # class_weight=class_weight,  # 일단 주석 처리
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

##############################################################################
# 9. 최종 모델 저장
##############################################################################
model.save("model_v4.1")
print("Model saved to model_v4.1")