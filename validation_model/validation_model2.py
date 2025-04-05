import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import matplotlib.pyplot as plt


# 모델 및 데이터 저장
model_path = "embedding_model_v0.3.h5"
embeddings_path = "image_embeddings_v0.3.npy"
paths_path = "image_paths_v0.3.npy"

img_size = (224, 224)

# 이미지 전처리 함수
def preprocess_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

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
input_img_path = "/Users/dexterj/Desktop/model/testdata/testimg_combined.jpg"

print(f"Embeddings shape: {embeddings.shape}")  # (N, D)
print(f"Number of image paths: {len(image_paths)}")  # N
embeddings = embeddings[:len(image_paths)]
print(f"Updated embeddings shape: {embeddings.shape}")

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