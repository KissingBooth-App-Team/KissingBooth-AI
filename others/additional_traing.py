import os
import tensorflow as tf
from tensorflow.keras.models import load_model

##############################################################################
# 0. 설정
##############################################################################
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 5               # 추가 학습을 위한 epoch
LEARNING_RATE = 1e-4     # 낮은 learning rate로 미세 조정
MODEL_PATH = "model_file_v2.9.h5"

# 추가 학습을 위한 데이터 디렉토리
FINE_TUNE_NORMAL_DIR = "/Users/dexterj/Desktop/model/merge"  # 정상 이미지
FINE_TUNE_OUTLIER_DIR = "/Users/dexterj/Desktop/model/outlier"  # 이상치 이미지

##############################################################################
# 1. 데이터 준비
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

fine_tune_image_paths, fine_tune_labels = get_image_paths_and_labels(FINE_TUNE_NORMAL_DIR, FINE_TUNE_OUTLIER_DIR)

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0
    return image

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    return image

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

fine_tune_ds = create_dataset(fine_tune_image_paths, fine_tune_labels, BATCH_SIZE)

##############################################################################
# 2. 모델 불러오기 및 재학습 준비
##############################################################################
model = load_model(MODEL_PATH)
print("Model loaded from", MODEL_PATH)

# 미세 조정을 위해 optimizer를 다시 설정
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

##############################################################################
# 3. 추가 학습 (미세 조정)
##############################################################################
model.fit(
    fine_tune_ds,
    epochs=EPOCHS
)

##############################################################################
# 4. 수정된 모델 저장
##############################################################################
model.save("fine_tuned_model_v2.10.h5")
print("Fine-tuned model saved to fine_tuned_model_v2.9.h5")