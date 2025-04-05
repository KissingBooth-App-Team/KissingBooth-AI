from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 이미지와 모델 관련 파라미터
IMG_HEIGHT, IMG_WIDTH = 224, 224
MODEL_PATH = "embedding_model_v0.12.h5"  # 저장된 모델 파일 경로
TEST_IMAGE_PATH = "/Users/dexterj/Desktop/model/testdata/f154bf3d-01c2-466e-ac57-d24a79e32811_combined.jpg"  # 테스트할 이미지 파일 경로

# 1. 모델 로드
print("[INFO] Loading model from .h5 ...")
loaded_ae = tf.keras.models.load_model(MODEL_PATH)
loaded_ae.summary()

# 2. 이미지 불러오기 및 전처리
print(f"[INFO] Loading and preprocessing image from '{TEST_IMAGE_PATH}'...")
img = Image.open(TEST_IMAGE_PATH).convert('RGB')  # RGB로 변환 (투명도 제거)
img = img.resize((IMG_WIDTH, IMG_HEIGHT))
img_array = np.array(img).astype(np.float32) / 255.0  # [0,1] 범위로 스케일링

# 배치 차원 추가: 모델 입력은 배치 형태이어야 함
input_image = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

# 3. 모델로 예측 (재구성)
print("[INFO] Predicting reconstructed image ...")
reconstructed = loaded_ae.predict(input_image)

# 재구성 결과의 shape는 (1, 224, 224, 3)임
reconstructed_image = np.squeeze(reconstructed, axis=0)  # shape: (224, 224, 3)

# 4. 원본 이미지와 재구성 이미지 시각화
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(img_array)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(reconstructed_image)
axs[1].set_title('Reconstructed Image')
axs[1].axis('off')

plt.show()

# 5. 재구성 오차 계산 (선택 사항)
mse = np.mean((reconstructed_image - img_array) ** 2)
print(f"MSE for the provided image: {mse}")