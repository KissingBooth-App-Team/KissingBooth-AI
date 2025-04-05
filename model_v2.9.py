
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

##############################################################################
# 0. 하이퍼파라미터 & 설정
##############################################################################
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32       # 너무 큰 배치는 메모리 부담이 될 수 있으므로 조정
EPOCHS = 15           
LEARNING_RATE = 1e-4

# 데이터 디렉토리 설정
NORMAL_DIR = "/Users/dexterj/Desktop/model/merge"     # 정상 이미지 폴더
OUTLIER_DIR = "/Users/dexterj/Desktop/model/outlier"  # 이상치 이미지 폴더

##############################################################################
# 1. 데이터 준비: 이미지 경로와 라벨 수집
#    --> outlier=0, normal=1 로 라벨 변경
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
    
    # outlier=0, normal=1
    image_paths = outlier_paths + normal_paths
    labels = [0] * len(outlier_paths) + [1] * len(normal_paths)
    
    return image_paths, labels

all_image_paths, all_labels = get_image_paths_and_labels(NORMAL_DIR, OUTLIER_DIR)
print(f"Total images: {len(all_image_paths)}")
print(f"  Outlier(0): {all_labels.count(0)}, Normal(1): {all_labels.count(1)}")

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
    """임의의 색상 변형"""
    image = tf.image.random_brightness(image, max_delta=0.8 * s)
    image = tf.image.random_contrast(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    image = tf.image.random_saturation(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    image = tf.image.random_hue(image, max_delta=0.2 * s)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

@tf.function
def augment_image(image):
    """수평 뒤집기 + 컬러 변형"""
    image = tf.image.random_flip_left_right(image)
    image = color_jitter(image, s=0.5)
    return image

##############################################################################
# 3. 데이터셋 구성: (img, label) 형태
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
# 4. 모델 구성: Base Encoder + 이진분류(Dense(1, sigmoid))
##############################################################################
base_encoder = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    pooling='avg',        # Global Average Pooling
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# 분류 헤드
classification_head = models.Sequential([
    layers.Dense(1, activation='sigmoid')
], name="classification_head")

# 전체 모델
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_encoder(inputs, training=True)
outputs = classification_head(x)
model = tf.keras.Model(inputs, outputs, name="OutlierVsNormal")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

##############################################################################
# 5. 불균형 데이터 대비 (class_weight) 설정 (선택적)
#    - 예시로 outlier=0 이 적으므로 가중치 높게 부여
#    - 실제 값은 실험적으로 조정 가능
##############################################################################
num_outlier = all_labels.count(0)
num_normal = all_labels.count(1)
total = num_outlier + num_normal

# 간단 계산: class_weight = total / (2 * class_count)
w_outlier = total / (2.0 * num_outlier + 1e-8)  # 분모 0 방지
w_normal = total / (2.0 * num_normal + 1e-8)

class_weight = {0: w_outlier, 1: w_normal}
print("Class Weights:", class_weight)
# 필요 없으면 아래처럼 None 설정
# class_weight = None

##############################################################################
# 6. 학습
##############################################################################
model.fit(
    train_ds,
    epochs=EPOCHS,
    class_weight=class_weight  # 불균형 조정(필요시)
)

##############################################################################
# 7. 모델 저장
##############################################################################
model.save("model_file_v2.9.h5")
print("Model saved to model_file_v2.9.h5")

