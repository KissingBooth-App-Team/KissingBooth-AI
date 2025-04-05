import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# 데이터셋 폴더 경로
data_dir = "/Users/dexterj/Desktop/cropping/merge"
img_size = (224, 224)
batch_size = 32
epochs = 20  # 학습 에포크 수

# 이미지 전처리 함수
def preprocess_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# 데이터 로드
def load_images(data_dir):
    image_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    images = np.array([preprocess_image(img_path) for img_path in image_files])
    labels = np.ones(len(images))  # 모든 이미지는 같은 클래스(1)로 간주
    return images, labels

# 데이터 로드 및 학습-검증 분리
images, labels = load_images(data_dir)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# 데이터 증강
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = datagen.flow(X_val, y_val, batch_size=batch_size)

# MobileNetV2 기반 복잡한 임베딩 모델 생성
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 추가 레이어를 통해 복잡도 증가
x = Dense(1024, activation="relu")(x)  # 첫 번째 Dense 레이어 (1024 유닛)
x = BatchNormalization()(x)            # BatchNormalization
x = Dropout(0.3)(x)                    # Dropout으로 과적합 방지
x = Dense(1280, activation="relu")(x)  # 임베딩 차원을 유지하는 Dense 레이어 (1280 유닛)
x = BatchNormalization()(x)            # BatchNormalization
embedding_model = Model(inputs=base_model.input, outputs=x)

# 모델 컴파일
embedding_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 콜백 설정
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint("embedding_model_v0.4_best.h5", save_best_only=True, monitor="val_loss", mode="min")
]

# 모델 학습
history = embedding_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=callbacks
)

# 학습 완료 후 모델 저장
embedding_model.save("embedding_model_v0.4.h5")
print("Model training complete and saved to embedding_model_v0.4.h5")