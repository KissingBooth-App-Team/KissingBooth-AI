# inference_simclr.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

##############################################################################
# 0. 하이퍼파라미터 & 설정 (학습 때 썼던 것과 동일해야 함)
##############################################################################
IMG_HEIGHT = 224
IMG_WIDTH = 224

##############################################################################
# 1. 모델, 임베딩, 이미지 경로 로드
##############################################################################
# (1) 저장된 SimCLR 임베딩 모델 로드
embedding_model = tf.keras.models.load_model("embedding_model_v0.5.h5")
print("Embedding model loaded from embedding_model_v0.5.h5")

# (2) 전체 이미지 임베딩과 이미지 경로 로드
all_embeddings = np.load("all_embeddings.npy")  # shape: (num_images, embedding_dim)
all_image_paths = np.load("all_image_paths.npy", allow_pickle=True)  # list of image file paths
print(f"Embeddings shape: {all_embeddings.shape}")
print(f"Total images loaded: {len(all_image_paths)}")

##############################################################################
# 2. 이미지 전처리 함수 (학습 때와 동일해야 함)
##############################################################################
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0  # [0,1] 스케일 정규화
    return image

##############################################################################
# 3. 유사도 계산 함수
##############################################################################
def find_most_similar_image(input_img_path, k=1):
    """
    새로운 이미지 경로(input_img_path)에 대해,
    학습 시 사용했던 이미지들(all_image_paths) 중 가장 유사한 Top-K 이미지를 찾는 함수.
    """
    # 1) 입력 이미지 전처리 & 임베딩
    img = load_and_preprocess_image(input_img_path)
    img = tf.expand_dims(img, axis=0)  # (1, H, W, 3)
    
    # 모델 추론 -> 임베딩
    input_emb = embedding_model(img, training=False)
    # 코사인 유사도 계산 편의를 위해 L2 Normalize
    input_emb = tf.math.l2_normalize(input_emb, axis=1).numpy()  # (1, embedding_dim)

    # 2) 코사인 유사도 계산 (dot product, 이미 정규화했으므로 dot = cosine similarity)
    similarities = np.dot(input_emb, all_embeddings.T)  # (1 x num_images)

    # 3) 유사도 내림차순 정렬 후 상위 K개 인덱스 찾기
    top_k_indices = np.argsort(similarities[0])[::-1][:k]

    # 4) 결과 (파일 경로, 유사도) 형태로 반환
    return [(all_image_paths[i], similarities[0][i]) for i in top_k_indices]

##############################################################################
# 4. 테스트 (메인 실행 구간)
##############################################################################
if __name__ == "__main__":
    # 이 부분은 실제 테스트용 코드, 필요에 따라 수정
    test_image_path = "/Users/dexterj/Desktop/model/testdata/testimg_combined.jpg"  # 테스트할 새 이미지 경로
    top1 = find_most_similar_image(test_image_path, k=1)
    
    print("Test Image:", test_image_path)
    print("Top1 Most similar:", top1)

    # 시각화
    test_img_bgr = cv2.imread(test_image_path)
    test_img_rgb = cv2.cvtColor(test_img_bgr, cv2.COLOR_BGR2RGB)

    most_similar_path, sim_score = top1[0]
    similar_img_bgr = cv2.imread(most_similar_path)
    similar_img_rgb = cv2.cvtColor(similar_img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(test_img_rgb)
    plt.title("Test Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(similar_img_rgb)
    plt.title(f"Top1 Similar (score={sim_score:.4f})")
    plt.axis("off")
    plt.show()