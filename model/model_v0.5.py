import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. 데이터 로드 및 전처리
# ----------------------------------------
# 데이터셋 폴더 경로
data_dir = "/Users/dexterj/Desktop/cropping/merge"  # 데이터셋 폴더 경로
img_size = (224, 224)

# 이미지 전처리 함수
def preprocess_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# 데이터셋 로드
def load_dataset(data_dir):
    image_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    images = np.array([preprocess_image(img_path) for img_path in image_files])
    return images, image_files

# 데이터셋 로드
images, image_files = load_dataset(data_dir)

# ----------------------------------------
# 2. 임베딩 모델 생성
# ----------------------------------------
# MobileNetV2 기반 임베딩 모델 생성
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
embedding_model = Model(inputs=base_model.input, outputs=x)

# 데이터셋 임베딩 생성
print("Generating embeddings for dataset...")
embeddings = embedding_model.predict(images)
print(f"Generated embeddings for {len(embeddings)} images.")

# ----------------------------------------
# 3. 저장 및 로드
# ----------------------------------------
# 모델 및 임베딩 저장
def save_model_and_embeddings(embedding_model, embeddings, image_paths, model_path, embeddings_path, paths_path):
    # 모델 저장
    embedding_model.save(model_path)
    print(f"Model saved to {model_path}")

    # 임베딩과 이미지 경로 저장
    np.save(embeddings_path, embeddings)
    np.save(paths_path, np.array(image_paths))
    print(f"Embeddings saved to {embeddings_path}")
    print(f"Image paths saved to {paths_path}")

# 모델 및 데이터 저장
model_path = "embedding_model.h5"
embeddings_path = "image_embeddings.npy"
paths_path = "image_paths.npy"
save_model_and_embeddings(embedding_model, embeddings, image_files, model_path, embeddings_path, paths_path)

# 저장된 모델 및 임베딩 로드
def load_model_and_embeddings(model_path, embeddings_path, paths_path):
    # 모델 로드
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")

    # 임베딩 및 이미지 경로 로드
    embeddings = np.load(embeddings_path)
    image_paths = np.load(paths_path, allow_pickle=True)
    print(f"Embeddings loaded from {embeddings_path}")
    print(f"Image paths loaded from {paths_path}")
    return model, embeddings, image_paths

# 모델 및 데이터 로드
embedding_model, embeddings, image_paths = load_model_and_embeddings(model_path, embeddings_path, paths_path)

# ----------------------------------------
# 4. 유사도 계산 함수
# ----------------------------------------
# 유사도 계산
def calculate_similarity(input_img_path, embeddings, image_paths):
    # 입력 이미지 전처리
    input_img = preprocess_image(input_img_path)
    input_img = np.expand_dims(input_img, axis=0)

    # 입력 이미지 임베딩 생성
    input_embedding = embedding_model.predict(input_img)

    # 코사인 유사도 계산
    similarities = cosine_similarity(input_embedding, embeddings)
    most_similar_idx = np.argmax(similarities)  # 가장 높은 유사도를 가진 인덱스

    # 결과 반환
    most_similar_image_path = image_paths[most_similar_idx]
    similarity_score = similarities[0, most_similar_idx]
    return most_similar_image_path, similarity_score

# ----------------------------------------
# 5. 테스트
# ----------------------------------------
# 입력 이미지 경로
input_img_path = "testimg_combined2.jpg"

# 유사도 계산
most_similar_image_path, similarity_score = calculate_similarity(input_img_path, embeddings, image_paths)

# 결과 출력
print(f"Input image: {input_img_path}")
print(f"Most similar image: {most_similar_image_path}")
print(f"Similarity score: {similarity_score:.4f}")

# 가장 유사한 이미지 시각화
input_img = cv2.imread(input_img_path)
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

most_similar_img = cv2.imread(most_similar_image_path)
most_similar_img = cv2.cvtColor(most_similar_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(input_img)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(most_similar_img)
plt.title(f"Most Similar (Score: {similarity_score:.4f})")
plt.axis("off")
plt.show()