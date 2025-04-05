#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example: Saving a "Pretrained-Encoder + Custom-Decoder" Autoencoder
into a SavedModel format, then re-loading it and testing reconstruction.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def main():
    # -------------------------------------------------------------
    # 1. 파라미터 설정
    # -------------------------------------------------------------
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    LATENT_DIM = 128
    EPOCHS = 2
    BATCH_SIZE = 4
    
    # -------------------------------------------------------------
    # 2. 사전학습된 모델(Encoder) 정의
    # -------------------------------------------------------------
    pretrained_backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        pooling='avg',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    encoder_inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='encoder_input')
    x = pretrained_backbone(encoder_inputs)
    latent = layers.Dense(LATENT_DIM, activation='relu', name='latent_dense')(x)
    encoder = models.Model(encoder_inputs, latent, name="pretrained_encoder")

    # -------------------------------------------------------------
    # 3. Decoder 정의 (ConvTranspose)
    # -------------------------------------------------------------
    decoder_inputs = layers.Input(shape=(LATENT_DIM,), name='decoder_input')
    init_size = 7
    num_filters = 128
    
    d = layers.Dense(init_size * init_size * num_filters, activation='relu')(decoder_inputs)
    d = layers.Reshape((init_size, init_size, num_filters))(d)
    d = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(d)
    d = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(d)
    d = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(d)
    d = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(d)
    d = layers.Conv2DTranspose(8, 3, strides=2, padding='same', activation='relu')(d)
    decoded_output = layers.Conv2D(3, kernel_size=3, padding='same', activation='sigmoid')(d)

    decoder = models.Model(decoder_inputs, decoded_output, name="custom_decoder")

    # -------------------------------------------------------------
    # 4. Autoencoder 구성 (Encoder + Decoder)
    # -------------------------------------------------------------
    autoencoder_inputs = encoder_inputs
    autoencoder_latent = encoder(autoencoder_inputs)
    autoencoder_outputs = decoder(autoencoder_latent)
    autoencoder = models.Model(autoencoder_inputs, autoencoder_outputs, name="autoencoder_pretrained")

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()

    # -------------------------------------------------------------
    # 5. 임의 데이터 생성 & 학습
    # -------------------------------------------------------------
    N = 12
    fake_data = np.random.rand(N, IMG_HEIGHT, IMG_WIDTH, 3).astype(np.float32)

    print("\n[INFO] Fitting on random data ...")
    autoencoder.fit(
        x=fake_data,
        y=fake_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    # -------------------------------------------------------------
    # 6. SavedModel 형식으로 모델 저장
    # -------------------------------------------------------------
    
    autoencoder.save("embedding_model_v0.12.h5")
if __name__ == "__main__":
    main()