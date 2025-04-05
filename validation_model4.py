import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

##############################################################################
# 0. 설정 (학습 시 사용한 것과 동일해야 함)
##############################################################################
IMG_HEIGHT = 224
IMG_WIDTH = 224

# 파일 경로 설정 (필요에 따라 수정)
MODEL_PATH = "/Users/dexterj/Desktop/model/embedding_model_v2.5.h5"       # 저장된 임베딩 모델 경로
EMBEDDINGS_PATH = "/Users/dexterj/Desktop/model/embedding_model_embedding_v2.5.npy"             # 저장된 임베딩 데이터 경로
IMAGE_PATHS_PATH = "/Users/dexterj/Desktop/model/embedding_model_image_path_v2.5.npy"           # 저장된 이미지 경로 리스트 경로

# 테스트할 새로운 이미지 경로
TEST_IMAGE_PATH = "/Users/dexterj/Desktop/model/testdata/gay.jpg" # 실제 테스트할 이미지 파일 경로로 변경

##############################################################################
# 1. 모델 및 임베딩 데이터 로드
##############################################################################
# 1-1. 저장된 임베딩 모델 로드
try:
    embedding_model = tf.keras.models.load_model(MODEL_PATH)
    print(f"임베딩 모델이 '{MODEL_PATH}' 에서 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    exit(1)

# 1-2. 전체 이미지 임베딩과 이미지 경로 로드
try:
    all_embeddings = np.load(EMBEDDINGS_PATH)
    all_image_paths = np.load(IMAGE_PATHS_PATH, allow_pickle=True)
    print(f"임베딩 데이터가 '{EMBEDDINGS_PATH}' 에서 로드되었습니다.")
    print(f"이미지 경로 리스트가 '{IMAGE_PATHS_PATH}' 에서 로드되었습니다.")
    print(f"임베딩 형태: {all_embeddings.shape}")
    print(f"전체 이미지 수: {len(all_image_paths)}")
except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e}")
    exit(1)
except Exception as e:
    print(f"임베딩 데이터 로드 중 오류 발생: {e}")
    exit(1)

##############################################################################
# 2. 이미지 전처리 함수 (학습 때와 동일하게 정의)
##############################################################################
def load_and_preprocess_image(path):
    """하나의 이미지 파일 경로를 읽어와, 학습 때와 같은 방식으로 전처리"""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0  # [0,1] 범위로 정규화
    return image

##############################################################################
# 3. 유사도 계산 함수
##############################################################################
def find_most_similar_image(input_img_path, k=1):
    """
    새로운 이미지 경로(input_img_path)에 대해,
    학습 시 사용했던 이미지들(all_image_paths) 중 가장 유사한 Top-K 이미지를 찾는 함수.
    
    Args:
        input_img_path (str): 테스트할 새로운 이미지의 파일 경로
        k (int): 찾고자 하는 유사한 이미지의 수 (기본값: 1)
    
    Returns:
        list of tuples: [(유사 이미지 경로, 유사도 점수), ...] 형태의 리스트
    """
    # 1) 입력 이미지 전처리 & 임베딩
    img = load_and_preprocess_image(input_img_path)
    img = tf.expand_dims(img, axis=0)  # (1, H, W, 3)
    
    # 모델 추론 -> 임베딩
    input_emb = embedding_model(img, training=False)
    # 코사인 유사도 계산 편의를 위해 L2 정규화
    input_emb = tf.math.l2_normalize(input_emb, axis=1).numpy()  # (1, embedding_dim)

    # 2) 코사인 유사도 계산 (dot product, 이미 정규화했으므로 dot = cosine similarity)
    similarities = np.dot(input_emb, all_embeddings.T)  # (1, num_images)

    # 3) 유사도 내림차순 정렬 후 상위 K개 인덱스 찾기
    top_k_indices = np.argsort(similarities[0])[::-1][:k]

    # 4) 결과 (파일 경로, 유사도) 형태로 반환
    return [(all_image_paths[i], similarities[0][i]) for i in top_k_indices]

##############################################################################
# 4. 시각화 함수
##############################################################################
def visualize_similar_images(test_img_path, similar_images):
    """
    테스트 이미지와 유사한 이미지를 시각화하는 함수.
    
    Args:
        test_img_path (str): 테스트할 이미지 파일 경로
        similar_images (list of tuples): [(유사 이미지 경로, 유사도 점수), ...] 형태의 리스트
    """
    # 테스트 이미지 로드 & 변환 (BGR -> RGB)
    test_img_bgr = cv2.imread(test_img_path)
    if test_img_bgr is None:
        print(f"테스트 이미지를 로드할 수 없습니다: {test_img_path}")
        return
    test_img_rgb = cv2.cvtColor(test_img_bgr, cv2.COLOR_BGR2RGB)

    # 유사 이미지 로드 & 변환 (BGR -> RGB)
    most_similar_path, sim_score = similar_images[0]
    similar_img_bgr = cv2.imread(most_similar_path)
    if similar_img_bgr is None:
        print(f"유사 이미지를 로드할 수 없습니다: {most_similar_path}")
        return
    similar_img_rgb = cv2.cvtColor(similar_img_bgr, cv2.COLOR_BGR2RGB)

    # 시각화
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(test_img_rgb)
    plt.title("테스트 이미지")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(similar_img_rgb)
    plt.title(f"가장 유사한 이미지\n유사도 점수: {sim_score:.4f}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

##############################################################################
# 5. 메인 실행 구간
##############################################################################
if __name__ == "__main__":
    # 테스트할 새로운 이미지 경로 설정
    test_image_path = TEST_IMAGE_PATH  # 실제 테스트할 이미지 파일 경로로 변경

    if not os.path.exists(test_image_path):
        print(f"테스트 이미지 파일이 존재하지 않습니다: {test_image_path}")
        exit(1)

    # 1) 가장 유사한 이미지 찾기
    top1 = find_most_similar_image(test_image_path, k=1)
    print("테스트 이미지:", test_image_path)
    print("가장 유사한 이미지:", top1)

    # 2) 시각화
    visualize_similar_images(test_image_path, top1)