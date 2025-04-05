import tensorflow as tf

def convert_to_tflite_select_tf_ops(h5_model_path, tflite_output_path):
    """
    Keras 모델(.h5)을 TensorFlow Lite 모델(.tflite)로 변환할 때
    Select TF Ops(Flex ops)를 활성화하여 변환하는 예시 함수.
    
    :param h5_model_path: 변환할 .h5 모델 경로 (str)
    :param tflite_output_path: 변환된 .tflite 모델 파일을 저장할 경로 (str)
    """
    # 1) Keras 모델 로드
    model = tf.keras.models.load_model(h5_model_path)

    # 2) TFLiteConverter 생성
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 3) Select TF Ops 사용 설정
    #    - TFLITE_BUILTINS(기본 TFLite 연산) + SELECT_TF_OPS(미지원 연산을 TF Select로 처리)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # (선택) 최적화 옵션 예시
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 4) 변환 수행
    tflite_model = converter.convert()

    # 5) .tflite 파일로 저장
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite 변환 완료! → {tflite_output_path}")

# 실제 실행 예시
if __name__ == "__main__":
    h5_model_path = "model_file_v2.9.h5"
    tflite_output_path = "model_file_v2.9_select_ops.tflite"
    
    convert_to_tflite_select_tf_ops(h5_model_path, tflite_output_path)