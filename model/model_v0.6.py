import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터셋 폴더 경로
data_dir = "/Users/dexterj/Desktop/cropping/merge"
img_size = (224, 224)

# 이미지 전처리 함수
def preprocess_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# 데이터셋 로드 및 증강
def load_and_augment_dataset(data_dir):
    image_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    images = np.array([preprocess_image(img_path) for img_path in image_files])
    
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
    
    augmented_images = []
    for img in images:
        img = np.expand_dims(img, axis=0)
        for _ in range(5):  # 이미지당 5개의 증강 이미지 생성
            augmented_images.append(datagen.flow(img, batch_size=1)[0][0])
    augmented_images = np.array(augmented_images)
    
    return augmented_images, image_files

# 데이터셋 로드
images, image_files = load_and_augment_dataset(data_dir)

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

# 데이터셋 임베딩 생성
print("Generating embeddings for dataset...")
embeddings = embedding_model.predict(images)
print(f"Generated embeddings for {len(embeddings)} images. Embedding shape: {embeddings.shape}")

# 모델 및 임베딩 저장 함수
def save_model_and_embeddings(embedding_model, embeddings, image_paths, model_path, embeddings_path, paths_path):
    embedding_model.save(model_path)
    np.save(embeddings_path, embeddings)
    np.save(paths_path, np.array(image_paths))
    print(f"Model saved to {model_path}")
    print(f"Embeddings saved to {embeddings_path}")
    print(f"Image paths saved to {paths_path}")

# 저장
model_path = "embedding_model_v0.3.h5"
embeddings_path = "image_embeddings_v0.3.npy"
paths_path = "image_paths_v0.3.npy"
save_model_and_embeddings(embedding_model, embeddings, image_files, model_path, embeddings_path, paths_path)